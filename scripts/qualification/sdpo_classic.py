"""Arms 2/3 (plan of 15 Nov; Ali 15 Nov: train arms 2 and 3 once the feedback is verified): classic SDPO, one round.

Data: the start model's own recorded attempts at the `--fit` tasks (the training half H1), from `--a3-episodes` (W6
for arm 2 = the SFT model; the base-model run for arm 3); only failed attempts, since a success has nothing to fix.
Teacher: a frozen copy of the start model (`sdpo_teacher`, never updated) reading the attempt with the chosen feedback
version (`--classic-version` N or P, feedback_versions.py, built live from the engine's T-A2 context, identical to the
measured text). Student: the start model (`--adapter`; none = the base model) on its own tokens.
Loss: the runner's per-token top-k + tail reverse KL toward the teacher (`episode_loss`, importance-weighted against the
recorded sampling log-probabilities), on reasoning and tool-call tokens (`--loss-tokens reasoning+calls`, arms 2/3) or on every assistant token,
visible text included (`--loss-tokens all`, like SFT: Ali 16 Nov for the full run). AdamW `--lr`, `--classic-batch` attempts per optimizer step, gradient clip 1.0, one pass.
`--classic-limit N --classic-longest-first` = the memory proof (the N longest attempts, nothing saved but the log).
`--a3-items` = the CPU-audited items (episode_id, digest of the version's teacher context, built by the same
`teacher_context` on the laptop): only those attempts train, and an attempt whose rebuilt context has another digest
is refused (counted as digest_mismatch; the queue fails the run on any). Writes classic-adapter/ (the trained student), classic-summary.json, classic-steps.jsonl, classic-admission.jsonl.
"""
from __future__ import annotations

import json

LIVE_VERSIONS = ("N", "P")
# 17 Nov (plan Step 5): the engine versions of Step 3 (V1 ... V4-32) are read from the committed, shipped rewrites file
# by the digest of the T-A2 context (sdpo_a3.teacher_context, arm "T-<version>"), exactly as the solve tests read them
REWRITE_VERSIONS = ("V1", "V2", "V3-1", "V3-2", "V4-1", "V4-2", "V4-31", "V4-32", "V5")  # V5: 17 Nov morning
VERSIONS = LIVE_VERSIONS + REWRITE_VERSIONS
LOSS_TOKENS = ("reasoning+calls", "all")  # "all" = reasoning, calls and visible text, like SFT (Ali, 16 Nov)


def make_engines(version):
    """(hidden, shown): the T-A2 engine and, for version P, the values-shown engine its pointers come from."""
    import feedback_versions as fv
    from sdpo_a3 import ENGINE_OPTIONS, make_engine

    shown = None
    if version in REWRITE_VERSIONS:
        return make_engine("T-A2"), None
    if version == "P":
        from sdpo_ab_engine import EngineContext

        shown = EngineContext(expected_values=True, pool=False, fix_hints=list(fv.SHOWN_ENGINE["fix_hints"]),
                              options=dict(ENGINE_OPTIONS), withhold_untrue=True)
    return make_engine("T-A2"), shown


def teacher_context(record, siblings, version, hidden, shown):
    """(the teacher's context in `version`, or None when the engine has nothing to say; pointer counts). The same
    function builds the CPU-audited digests (classic_prepare) and the GPU training contexts."""
    import feedback_versions as fv

    context = hidden(record, siblings, "A", "both")
    if context is None:
        return None, {}
    if version in REWRITE_VERSIONS:
        import sdpo_a3

        return sdpo_a3.teacher_context("T-" + version, context), {}
    if version == "N":
        return fv.version_n(context), {}
    return fv.version_p(context, shown(record, siblings, "A", "both"))


def admitted_attempts(records, outcomes, fit):
    """The failed, scored attempts of the fit tasks (an infrastructure failure or a success is not trained)."""
    import sdpo_ab_world as world

    out = []
    for record in records.values():
        o = outcomes.get(record["id"])
        if world.task_name(record) in fit and o and not o.get("infrastructure_failure") and o["success"] != 1:
            out.append(record)
    return out


async def phase_classic(rt, args, runner):
    import asyncio
    import concurrent.futures
    import gzip
    import random
    from collections import Counter, defaultdict
    from pathlib import Path

    from peft import load_peft_weights, set_peft_model_state_dict

    import sdpo_ab_world as world
    from sdpo_a3 import context_digest

    if not runner.THINKING:
        raise ValueError("--phase classic-train needs --thinking on")
    if args.loss_tokens not in LOSS_TOKENS:
        raise ValueError(f"--phase classic-train needs --loss-tokens in {LOSS_TOKENS}")
    if args.classic_version not in VERSIONS:
        raise ValueError(f"--classic-version must be one of {VERSIONS}")
    start = load_peft_weights(args.adapter) if args.adapter else load_peft_weights(str(rt.initial))
    for name in ("default", "sdpo_teacher"):
        set_peft_model_state_dict(rt.model, start, adapter_name=name)
    rt.model.set_adapter("default")
    fit = frozenset(json.loads(Path(args.fit).read_text())["task_names"])
    run = Path(args.a3_episodes)
    outcomes = {o["episode_id"]: o for o in json.loads((run / "outcomes.json").read_text())}
    records, chains = {}, {}
    with gzip.open(run / "episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if world.task_name(r) in fit:
                records[r["id"]] = r
    with gzip.open(run / "chains.jsonl.gz", "rt") as stream:
        for line in stream:
            c = json.loads(line)
            if c["episode_id"] in records:
                chains[c["episode_id"]] = c
    by_task = defaultdict(list)
    for record in records.values():
        by_task[world.task_name(record)].append(record)
    if not args.a3_items:
        raise ValueError("--phase classic-train needs --a3-items (the CPU-audited teacher-context digests)")
    audited = {}
    for line in Path(args.a3_items).read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("version") != args.classic_version:
                raise ValueError(f"audited items are for version {item.get('version')}, not {args.classic_version}")
            audited[item["episode_id"]] = item["digest"]
    attempts = [r for r in admitted_attempts(records, outcomes, fit) if r["id"] in audited]
    rng = random.Random(20261115)
    rng.shuffle(attempts)
    if args.classic_longest_first:
        attempts.sort(key=lambda r: -sum(len(s["ids"]) for s in (chains.get(r["id"]) or {"segments": []})["segments"]))
    if args.classic_limit:
        attempts = attempts[: args.classic_limit]
    hidden, shown = make_engines(args.classic_version)
    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="classic-engine")
    admission, prepared, counts, versions = [], [], Counter(), Counter()
    for record in attempts:
        entry = {"episode_id": record["id"], "task": world.task_name(record)}
        chain = chains.get(record["id"])
        if chain is None or chain.get("mismatch"):
            admission.append({**entry, "admitted": False, "reason": "chain_not_joined"})
            continue
        siblings = by_task[entry["task"]]
        try:
            context, made = await loop.run_in_executor(worker, teacher_context, record, siblings,
                                                       args.classic_version, hidden, shown)
            versions.update(made)
        except Exception as error:  # noqa: BLE001 - an engine refusal or crash excludes the attempt, never stops
            admission.append({**entry, "admitted": False, "reason": f"engine: {type(error).__name__}: {error}"[:300]})
            continue
        if context is None:
            admission.append({**entry, "admitted": False, "reason": "no_feedback"})
            continue
        if context_digest(context) != audited[record["id"]]:
            admission.append({**entry, "admitted": False, "reason": "digest_mismatch"})
            continue
        sequences = runner.prepare_sequences(rt.tokenizer, chain, context, args.loss_tokens)
        if isinstance(sequences, str):
            admission.append({**entry, "admitted": False, "reason": sequences})
            continue
        admission.append({**entry, "admitted": True, "tokens": len(sequences["student"]["targets"]),
                          "teacher_length": len(sequences["teacher"]["ids"])})
        prepared.append((entry, sequences))
    for a in admission:
        counts["admitted" if a["admitted"] else "excluded:" + a["reason"].split(":")[0]] += 1
    runner.log(stage="classic_prepared", version=args.classic_version, attempts=len(attempts), **counts)
    batch = max(1, args.classic_batch)
    steps, losses, oom, peak = train_pass(rt.model, prepared, runner.episode_loss, args.lr, batch,
                                          rt.out / "classic-steps.jsonl", runner.log)
    rt.model.set_adapter("default")
    if not args.classic_limit:
        rt.model.save_pretrained(rt.out / "classic-adapter", selected_adapters=["default"], safe_serialization=True)
    with open(rt.out / "classic-admission.jsonl", "w") as stream:
        for a in admission:
            stream.write(json.dumps(a) + "\n")
    summary = {"version": args.classic_version, "audited_items": len(audited), "start_adapter": args.adapter, "episodes": str(run), "fit": args.fit,
               "lr": args.lr, "batch": batch, "attempts": len(attempts), "trained": len(losses), "oom": oom,
               "steps": len(steps), "peak_mib": peak, "counts": dict(counts), "pointer_counts": dict(versions),
               "first_loss": steps[0]["loss"] if steps else None, "last_loss": steps[-1]["loss"] if steps else None,
               "memory_proof": bool(args.classic_limit), "saved": not args.classic_limit}
    (rt.out / "classic-summary.json").write_text(json.dumps(summary, indent=1))
    runner.log(stage="classic_done", **{k: v for k, v in summary.items() if k != "counts"})


def train_pass(model, prepared, episode_loss, lr, batch, steps_path, log, peak_mib=None, after_step=None,
               optimizer=None):
    """One pass over `prepared` [(entry, sequences)]: AdamW on the trainable parameters, `batch` attempts per step,
    gradient clip 1.0; an attempt that runs out of memory is skipped and counted. `episode_loss` returns (loss,
    per-token importance weights); a step's "ratio" is the mean of its attempts' mean weights.
    SDPO v6 (19 Nov): `after_step` is called after every optimizer step (the online phase's EMA teacher) and
    `optimizer` lets the online rounds keep one AdamW across rounds; both default to the classic behaviour.
    Returns (steps, losses, oom, peak MiB)."""
    import time

    import torch

    if peak_mib is None:
        peak_mib = lambda: torch.cuda.max_memory_allocated() // 2**20  # noqa: E731
    parameters = [p for _, p in model.named_parameters() if p.requires_grad]
    if optimizer is None:
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    steps, losses, ratios, oom, peak = [], [], [], 0, 0
    started = time.monotonic()
    with open(steps_path, "w") as log_file:
        for index, (entry, sequences) in enumerate(prepared, start=1):
            try:
                loss, weights = episode_loss(model, sequences["student"], sequences["teacher"])
                (loss / batch).backward()
                losses.append(float(loss.detach()))
                if weights:
                    ratios.append(sum(weights) / len(weights))
            except torch.OutOfMemoryError:
                oom += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log(stage="classic_oom", **entry)
            peak = max(peak, peak_mib())
            if index % batch == 0 or index == len(prepared):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if after_step is not None:
                    after_step()
                recent = ratios[-batch:]
                row = {"step": len(steps) + 1, "attempts": index, "loss": sum(losses[-batch:]) / max(1, len(losses[-batch:])),
                       "ratio": sum(recent) / len(recent) if recent else None, "grad_norm": float(norm),
                       "oom": oom, "peak_mib": peak, "seconds": round(time.monotonic() - started, 1)}
                steps.append(row)
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                log(stage="classic_step", **row)
    return steps, losses, oom, peak
