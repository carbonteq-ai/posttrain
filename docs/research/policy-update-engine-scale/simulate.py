"""Replay a training round's post-collection path on real retained episodes.

Stages (each timed): load and validate native episodes, project them into
EnvironmentRollouts through the real Verifiers bridge, select spread groups the
way active collection does, retain the population artifact, admit it
(population snapshot, SAMPO credit, objective, updates, pack plans), build the
TRL population view and sampled scores, collection telemetry and, with
``--train``, reference scores and every optimizer update with a stand-in model.

See README.md in this directory for setup and the recorded baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import copyreg
import cProfile
import json
import math
import pickle
import pstats
import resource
import shutil
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ID = "manifest-steps-26-sampo-100-g16x8-20261005-r6"
ENVIRONMENT = "automationbench-manifest-steps-mixv2-4k16t-v1"
SETTINGS = "lfm2.5-2.6b/automationbench-manifest-steps-sampo-100-g16x8-v1"
PROJECTION = "reward/automationbench-manifest-steps@1"
INFERENCE = "inference/lfm2.5-2.6b-vllm-manifest-ws-c180-4k-t08-fp16@1"


def _proxy(mapping):
    return types.MappingProxyType(mapping)


copyreg.pickle(types.MappingProxyType, lambda m: (_proxy, (dict(m),)))

timings: dict[str, float] = {}
profile_stages: set[str] = set()
profile_dir: Path | None = None


@contextmanager
def stage(name: str):
    profiler = cProfile.Profile() if name in profile_stages else None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_start = usage.ru_utime + usage.ru_stime
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        yield
    finally:
        if profiler:
            profiler.disable()
            assert profile_dir is not None
            profiler.dump_stats(profile_dir / f"{name}.prof")
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative").print_stats(35)
        timings[name] = time.perf_counter() - start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = usage.ru_utime + usage.ru_stime - cpu_start
        rss = usage.ru_maxrss / 1e6
        print(
            f"[stage] {name:<28} {timings[name]:8.2f}s  main-process CPU {cpu:8.2f}s  peak RSS {rss:5.1f} GB",
            flush=True,
        )


def resolve(catalog, family: str, selection: str):
    from posttrain.common import CatalogRef

    return catalog.resolve(CatalogRef(family, selection)).value


def main() -> None:
    global profile_stages, profile_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", default=[], help="cProfile a stage by name (repeatable)")
    parser.add_argument("--rebuild", action="store_true", help="project the raw episodes even when a cache exists")
    parser.add_argument("--concurrency", type=int, default=32, help="episodes projected at once")
    parser.add_argument("--record-encoding", choices=("thread", "process"), default="process")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--train", action="store_true", help="also run every optimizer step with a stand-in model")
    parser.add_argument("--data", type=Path, required=True, help="directory with the run's episodes.jsonl")
    parser.add_argument("--project", type=Path, required=True, help="posttrain project root (apps/lab)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.episodes = args.data / "episodes.jsonl"
    args.cache = args.data / "population.pkl"
    args.projected = args.data / "episodes-projected.jsonl"
    profile_stages = set(args.profile)
    profile_dir = args.out
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    with stage("imports+catalog"):
        from posttrain.catalog import discover_project, open_catalog
        from posttrain.train.backends.policy_update_admission import AdmittedNativePopulation
        from posttrain.train.backends.trl.policy_job import SCORE_CONTRACT
        from posttrain.train.backends.trl.policy_updates import ResolvedTRLPopulation
        from posttrain.train.integrations.verifiers import create_verifiers_training_bridge
        from posttrain.train.online_rl import BehaviorPolicySpan
        from posttrain.train.profiles import shape_online_reward
        from posttrain.train.rollout_execution import CollectionKey, EpisodeKey
        from posttrain.train.update_plan import ExecutionCapabilities
        from posttrain.train.update_records import PolicyVersions
        from posttrain.train.update_telemetry import collection_metrics
        from posttrain.train.verifiers_requests import validate_verifiers_policy_sampling
        from verifiers.v1._validation_scope import validation_scope
        from verifiers.v1.episode import Episode

        layout = discover_project(args.project)
        catalog = open_catalog(
            scope=layout.project_id,
            overlays=layout.catalog_overlays,
            catalog_root=layout.base_catalog,
            required_plugin_distributions=layout.catalog_plugin_requirements,
        )
        environment = resolve(catalog, "environment", ENVIRONMENT)
        settings = resolve(catalog, "training", SETTINGS)
        projection = resolve(catalog, "training", PROJECTION)
        inference = resolve(catalog, "inference", INFERENCE)

    with stage("bridge"):
        sampling = validate_verifiers_policy_sampling(environment, inference, settings.max_completion_length)
        bridge = create_verifiers_training_bridge(
            environment,
            args.out / "verifiers-traces.jsonl",
            RUN_ID,
            sampling=sampling,
            purpose="sampo",
            reward_projection=projection,
            policy_update_context_contract="causal-text@1",
        )
        bridge.record_encoding = args.record_encoding

    if args.cache.exists() and not args.rebuild:
        with stage("load cached population"):
            population = pickle.loads(args.cache.read_bytes())
            shutil.copyfile(args.projected, args.out / "episodes.jsonl")
    else:
        with stage("load+validate episodes"):
            lines = args.episodes.read_bytes().splitlines()
            episodes = []
            for line in lines:
                with validation_scope():
                    episodes.append(Episode.model_validate_json(line))
        print(f"episodes {len(episodes)} ({sum(map(len, lines)) / 1e9:.2f} GB)")

        collection = CollectionKey(RUN_ID, "collection-0", "actor-0", 0)

        async def project_all():
            # The job keeps many episodes in flight; project them concurrently.
            limit = asyncio.Semaphore(args.concurrency)

            async def project(ordinal, episode):
                info = episode.traces[0].info
                key = EpisodeKey(
                    collection,
                    str(info["example_id"]),
                    str(info.get("posttrain_prompt_group_id") or info["example_id"]),
                    str(info.get("posttrain_rollout_id") or f"rollout-{ordinal}"),
                    ordinal,
                    ordinal,
                )
                async with limit:
                    return await bridge.project_native_episode(key, episode, behavior_policy=BehaviorPolicySpan(0, 0))

            return list(await asyncio.gather(*(project(ordinal, episode) for ordinal, episode in enumerate(episodes))))

        with stage("project episodes"):
            rollouts = asyncio.run(project_all())

        with stage("select groups"):
            groups: dict[str, list] = {}
            for rollout in rollouts:
                groups.setdefault(rollout.example_id, []).append(rollout)
            selected = []
            for group in groups.values():
                rewards = [
                    shape_online_reward(settings, r.reward, len(r.completion_ids), is_truncated=r.is_truncated)
                    for r in group
                ]
                if (
                    len(group) == settings.num_generations
                    and all(map(math.isfinite, rewards))
                    and max(rewards) > min(rewards)
                ):
                    selected.append(tuple(group))
            selected = selected[: args.groups]
            population = tuple(r for group in selected for r in group)
        if not args.cache.exists():
            args.cache.write_bytes(pickle.dumps(population, protocol=5))
            shutil.copyfile(args.out / "episodes.jsonl", args.projected)
    tokens = sum(len(r.completion_ids) for r in population)
    sampled = sum(sum(r.env_mask) for r in population)
    print(f"rollouts {len(population)} completion tokens {tokens} sampled tokens {sampled}")

    with stage("retain population"):
        artifact = bridge.retain_population(population)

    capabilities = ExecutionCapabilities(
        ("grpo@1", "dapo@1", "sampo@1", "sampo-spans@1", "gdpo@1", "capo@1"),
        ("sampled-logp", "old-logp", "reference-logp"),
        settings.max_prompt_length + settings.max_completion_length,
        True,
        prefix_sharing=True,
    )
    current = "sim/actor-0"
    with stage("admit (resolve+plan)"):
        admitted = AdmittedNativePopulation.from_retained_artifact(
            artifact,
            population,
            settings,
            capabilities,
            population_id=f"{RUN_ID}/population-at-0",
            template_revision="sim",
            versions=PolicyVersions(current, current, current, "sim/reference" if settings.beta else None),
            sampler_step=0,
            selector_digest="sim",
            applied_update_offset=0,
            attempt_offset=0,
            max_overflow_retries=0,
        )
    resolved = admitted.resolved
    print(
        f"actions {resolved.snapshot.size} contexts {len(resolved.snapshot.conditioning)} "
        f"updates {len(resolved.updates)} packs {[len(p) for p in resolved.packs]}"
    )

    with stage("trl population view"):
        view = ResolvedTRLPopulation.from_admitted(
            admitted, score_temperature=1.0, score_contract=SCORE_CONTRACT, sampler_correction=None
        )
        view.sampled_scores = admitted.read_input.sampling_log_scores(resolved.snapshot)

    with stage("collection telemetry"):
        collection_metrics(settings, population, credit_estimator_id=resolved.credit.estimator_id)

    if args.train:
        train_steps(args, settings, resolved, view, sampled_scores=view.sampled_scores)

    summary = {
        "timings": timings,
        "rollouts": len(population),
        "actions": resolved.snapshot.size,
        "update_digests": [u.digest for u in resolved.updates],
        "pack_digests": [[p.update_digest for p in packs][:1] for packs in resolved.packs],
        "credit_digest": resolved.credit.digest,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(timings, indent=1))


def train_steps(args, settings, resolved, view, *, sampled_scores) -> None:
    """Every optimizer step of this population through the real resolved loss path.

    The stand-in model replaces only the transformer and vocabulary math: each
    sampled token still becomes its own score tensor exactly as in the job, so
    the per-token bookkeeping, synchronizations and autograd graph are real.
    """
    from types import SimpleNamespace

    import torch
    from posttrain.train.backends.policy_update_scoring import freeze_population_scores
    from posttrain.train.backends.trl.policy_job import SCORE_CONTRACT
    from posttrain.train.update_sampler_correction import recipe_sampler_correction_weights

    device = torch.device("cuda")

    class StandIn(torch.nn.Module):
        """A tiny causal LM over the real 128,000-token vocabulary.

        The decoder is an embedding (no attention), so model compute is
        negligible while the engine's real scoring path runs: one decoder pass
        per cover and the chunked FP32 log-softmax over the full vocabulary at
        every sampled position, with real autograd into the parameters.
        """

        def __init__(self, vocabulary: int = 128_000, width: int = 8):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocabulary, width, device=device)
            self.head = torch.nn.Linear(width, vocabulary, bias=False, device=device)

        def decode(self, input_ids, **_):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        def get_decoder(self):
            return self.decode

        def get_output_embeddings(self):
            return self.head

    model = StandIn()
    snapshot = resolved.snapshot
    view.prepare_sampler_correction = lambda old: recipe_sampler_correction_weights(
        settings, snapshot, old, sampled_scores
    )
    if view.spec.beta:
        with stage("reference scores"):
            view.reference = freeze_population_scores(
                model,
                snapshot,
                read_input=view.read_input,
                device=device,
                policy_version=snapshot.versions.reference,
                score_contract=SCORE_CONTRACT,
                score_temperature=view.score_temperature,
                prefix_sharing=view.capabilities.prefix_sharing,
            )
    optimizer = SimpleNamespace()
    for index in range(len(view.updates)):
        with stage(f"update {index} loss"):
            loss = view.loss(model, index, device)
        with stage(f"update {index} backward"):
            loss.backward()
            torch.cuda.synchronize()
        view.before_step(optimizer)
        view.complete_step(optimizer)
        model.zero_grad(set_to_none=True)
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
