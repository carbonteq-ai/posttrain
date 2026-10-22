import json
from types import SimpleNamespace

import pytest
from preflight_sdpo_guidance_native import qualification_bootstrap


def test_qualification_population_is_explicitly_bound_not_renamed_as_training(tmp_path):
    prior = tmp_path / "step-0000/online_train"
    prior.mkdir(parents=True)
    receipt = tmp_path / "receipt.json"
    names = {f"t{i}" for i in range(20)}
    receipt.write_text(json.dumps({"qualified": True, "optimizer_updates": 0,
        "stochastic_native": {"qualified": True, "temperature": 1., "episodes": 40, "tasks": sorted(names)}}))
    args = SimpleNamespace(prior=prior, step=1, qualification_receipt=receipt)
    assert qualification_bootstrap(args, names, 40)
    with pytest.raises(ValueError, match="closed zero-update"):
        qualification_bootstrap(args, {"different"}, 40)
    args.step = 2
    with pytest.raises(ValueError, match="does not bind"):
        qualification_bootstrap(args, names, 40)


def test_existing_native_training_preflights_do_not_become_bootstrap_implicitly():
    assert not qualification_bootstrap(SimpleNamespace(), {"a"}, 40)
