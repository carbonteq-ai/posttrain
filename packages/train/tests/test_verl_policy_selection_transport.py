"""The isolated veRL host must preserve Posttrain's selected update meaning."""

from dataclasses import fields, replace
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.train.backends.verl.contracts import VerlAlgorithm, VerlPayload
from posttrain.train.backends.verl.launcher import grpo_algorithm_payload, sampo_algorithm_payload
from posttrain.train.backends.verl.worker import validate_policy_update_entrypoint
from posttrain.train.profiles import GRPOSettings, SAMPOSettings, TrainingLoop
from posttrain.train.requests import GRPORequest
from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings
from posttrain.train.update_records import ActionSelection, InvalidPolicyUpdate


@pytest.mark.parametrize("algorithm", ["grpo", "sampo"])
@pytest.mark.parametrize("variant", ["algorithm", "semantic-spans"])
def test_worker_json_roundtrip_preserves_schedule_masks_and_reduction(algorithm, variant):
    selected = PolicyUpdateSettings(
        PolicyUpdateSchedule("turn", 3, epochs=2, order="shuffle", seed=17, final_policy="include"),
        PolicyExecutionBudget(1, 4096, 100000),
    )
    if variant == "semantic-spans":
        selected = replace(selected, objective_variant=variant,
            policy_selection=ActionSelection("spans", ("thinking-1",)),
            kl_selection=ActionSelection("spans", ("answer-1",)),
            denominator="original-eligible", empty_policy="zero")
    if algorithm == "grpo":
        payload = grpo_algorithm_payload(GRPOSettings("transport", TrainingLoop(max_steps=3), policy_updates=selected))
    else:
        payload = sampo_algorithm_payload(SAMPOSettings("transport", TrainingLoop(max_steps=3), policy_updates=selected))
    contract = VerlAlgorithm.model_validate(payload)
    restored = VerlAlgorithm.model_validate_json(contract.model_dump_json())
    assert restored.policy_updates == selected
    assert restored.policy_updates is not None
    assert restored.policy_updates.schedule != PolicyUpdateSchedule("episode", 1)
    with pytest.raises(InvalidPolicyUpdate, match="legacy main_ppo cannot consume"):
        validate_policy_update_entrypoint(cast(VerlPayload, SimpleNamespace(algorithm=restored)))


def test_legacy_worker_selection_remains_absent():
    settings = GRPOSettings("legacy", TrainingLoop(max_steps=1, gradient_accumulation_steps=2),
                            num_prompts_per_step=1, num_generations=2)
    contract = VerlAlgorithm.model_validate(grpo_algorithm_payload(settings))
    assert contract.policy_updates is None
    validate_policy_update_entrypoint(cast(VerlPayload, SimpleNamespace(algorithm=contract)))


def test_worker_json_rejects_invalid_selected_budget():
    import json

    selection = PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2), PolicyExecutionBudget(1, 4096, 10000))
    settings = GRPOSettings("transport", TrainingLoop(max_steps=1), policy_updates=selection)
    payload = json.loads(VerlAlgorithm.model_validate(grpo_algorithm_payload(settings)).model_dump_json())
    payload["policy_updates"]["schedule"]["budget"] = 0
    with pytest.raises(ValueError, match="positive integers"):
        VerlAlgorithm.model_validate_json(json.dumps(payload))


def test_launcher_manifest_preserves_resolved_boundaries_without_legacy_batch_equality(tmp_path):
    from posttrain.train.backends.verl.contracts import VerlLaunchManifest
    from posttrain.train.backends.verl.launcher import build_grpo_launch_plan

    from .test_verl_backend import _grpo_request

    legacy = _grpo_request()
    # Internal candidate selection: public launch remains guarded until native
    # host qualification, while manifest transport can be checked independently.
    request = SimpleNamespace(**{field.name: getattr(legacy, field.name) for field in fields(legacy)})
    selection = PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1, epochs=2),
                                    PolicyExecutionBudget(1, 4096, 10000))
    request.settings = replace(legacy.settings,
        loop=replace(legacy.settings.loop, gradient_accumulation_steps=1), policy_updates=selection)
    manifest = build_grpo_launch_plan(cast(GRPORequest, request), tmp_path)
    restored = VerlLaunchManifest.model_validate_json(manifest.model_dump_json())
    assert restored.payload.algorithm.policy_updates == selection
    assert restored.payload.resolved_settings is not None
    assert restored.payload.resolved_settings.settings == request.settings
    assert restored.payload.training.loop.gradient_accumulation_steps == 1
    with pytest.raises(InvalidPolicyUpdate, match="resolved native population host"):
        validate_policy_update_entrypoint(restored.payload)
    import json

    drifted = json.loads(manifest.model_dump_json())
    drifted["payload"]["training"]["loop"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="native loop launch fields"):
        VerlLaunchManifest.model_validate_json(json.dumps(drifted))
    missing = json.loads(manifest.model_dump_json())
    del missing["payload"]["resolved_settings"]
    with pytest.raises(ValueError, match="complete typed settings"):
        VerlLaunchManifest.model_validate_json(json.dumps(missing))


@pytest.mark.parametrize("kind", ["grpo", "sampo", "gdpo", "capo"])
def test_full_settings_envelope_preserves_normalizer_values(kind):
    from posttrain.train.backends.verl.contracts import VerlResolvedSettings
    from posttrain.train.profiles import CAPOSettings, GDPOSettings
    from pydantic import TypeAdapter

    selection = PolicyUpdateSettings(PolicyUpdateSchedule("episode", 1, epochs=2),
                                    PolicyExecutionBudget(1, 4096, 10000))
    loop = TrainingLoop(max_steps=3, max_length=4096, warmup_ratio=.17, seed=97)
    if kind == "grpo":
        settings = GRPOSettings("original-settings", loop, policy_updates=selection,
                                algorithm="dapo", advantage_scaling="none")
    elif kind == "sampo":
        settings = SAMPOSettings("original-settings", loop, policy_updates=selection,
                                 discount_gamma=.87, step_advantage_weight=.3)
    elif kind == "gdpo":
        settings = GDPOSettings(id="original-settings", loop=loop, policy_updates=selection,
                                component_names=("task", "format"), component_weights=(1., .25))
    else:
        settings = CAPOSettings(id="original-settings", loop=loop, policy_updates=selection,
                                outcome_weight=3., process_weight=.75)
    adapter = TypeAdapter(VerlResolvedSettings)
    envelope = adapter.validate_python({"kind": kind, "settings": settings})
    restored = adapter.validate_json(adapter.dump_json(envelope))
    assert restored.settings == settings
    assert restored.settings.loop.max_length == 4096
    assert restored.settings.loop.warmup_ratio == .17
