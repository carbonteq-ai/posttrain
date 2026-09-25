# Agentic workload inference optimization with a frozen baseline

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. It follows `docs/templates/PLAN.md` from the repository root and must be maintained in accordance with it.

## Purpose / Big Picture

Posttrain trains and evaluates tool-using agents on AutomationBench. An agentic episode is many short model turns over one growing conversation: the model reads a system prompt, tool schemas and the conversation so far, emits a tool call of a few dozen to a few hundred tokens, the environment runs the tool, and the tool result is appended before the next turn. The earlier inference work in this repository optimized single-turn decode (see `docs/architecture/vllm-inference-optimization.md`). It never measured this shape of work, where the cost moves toward re-reading long histories (prefill), toward state kept between turns, and toward host-side overhead in the harness that drives the turns.

After this plan, a contributor can run one command that replays or executes a fixed set of the larger AutomationBench tasks against a chosen model on the RTX PRO 6000 workstation and prints, for a named configuration, episodes per hour, per-turn latency, time to first token, prefilled versus cached tokens, and where GPU and host time went. Every optimization in this plan is then reported as a speedup against one frozen baseline measured on the same tasks, seeds and hardware, together with a correctness guard. A speedup whose guard fails is reported as a failure. The results are also written to the Invariant Decode Ledger dashboard in a new "Agentic workloads" section.

## Progress

- [x] (2026-09-23 06:30Z) Surveyed AutomationBench tasks and tool execution, the rl rollout harness and serving configuration, and the vLLM fork's agentic features; findings are summarized in Context and Orientation.
- [x] (2026-09-23 06:40Z) Tool-result caching removed from scope at the user's request (see Decision Log).
- [x] (2026-09-23 07:05Z) Milestone 1 tooling: `replay.py`, `export_binding.py`, `make_lora.py` and `summarize.py` under `scripts/qualification/agentic_benchmark/`. All 355 recorded turns in the 50-episode sample rebuild with exactly the recorded prompt and output token counts (2 failed calls that exceeded the 24,576-token context are skipped). A 6-episode smoke run on the workstation completed: 326.5 episodes/h, 107 ms mean TTFT, 82.1% prefix-cache hit rate.
- [x] (2026-09-23 07:03Z) Milestone 1 measurement: frozen baseline, three repeats, 1,160-1,173 episodes/h (spread 1.1%); reference with prefix caching off, 1,057-1,060 episodes/h. Stock-defaults reference running.
- [x] (2026-09-23 07:52Z) Corrected-replay baselines (running context carried forward, collection mode). Production binding: c32 98.3-98.6 s per collection (about 1,170 episodes/h), 45.8% prefix hit, 49.4% of reusable context recomputed; c16 66.4 s per collection (868 episodes/h), 80.7% hit, 9.6% reusable context recomputed. Best-settings bundle at c16: 67.7 s on its only valid repeat (see Surprises), so the production binding is the best-settings reference.
- [x] (2026-09-23 07:59Z) Reuse-loss diagnosis: at low load every lost token is the previous turn's generated output (plus under one 16-token block). Cause and fix in Surprises; with dense Mamba retention the loss falls from 10.3% to 0.1% and prefilled tokens halve.
- [x] (2026-09-23 08:05Z) vLLM fork fix: decode continuation boundary in `KVCacheCoordinator.get_replay_boundaries` (`vllm/v1/core/kv_cache_coordinator.py`), with regression test `test_mamba_multi_turn_continuation_reuses_decoded_tokens` (fails without the fix: hit 0 instead of 80 tokens). Uncommitted in `/home/hammad/projects/vllm-sm120`. c16 replay (production binding, three identical repeats): 66.4 -> 64.9 s per collection (1.023x), prefilled tokens 486,770 -> 279,538 (-43%), recomputed reusable context 9.6% -> 0.1%, prefix hit 80.7% -> 88.9%, mean TTFT 115 -> 94 ms.
- [x] (2026-09-23 08:15Z) Milestone 8 startup: Verifiers fork server (`verifiers/v1/runtimes/zygote.py`, `_zygote_server.py`), port-file backoff and host-side probe (`verifiers/v1/mcp/launch.py`) on branch `codex/fast-rollout-startup` in `/home/hammad/projects/verifiers-fast-startup` (uncommitted). Mock-model live runs, 25 tasks x 2 rollouts at concurrency 16: host overhead per episode 5.88 s -> 0.42 s (setup 4.06 -> 0.12 s, harness 1.82 -> 0.30 s), wall 21.0 -> 4.3 s. Verifiers v1 suite: same 14 pre-existing environment-import failures before and after; 9 new fork-server tests pass.
- [x] (2026-09-23 08:30Z) Milestone 8 tool path: `live.py --replay-model` replays each recorded episode's assistant turns (tool calls included) through the real harness and tool server. Every MCP tool call took 46 ms, of which the server spent 3 ms; the rest was a Nagle/delayed-ACK stall (see Surprises), fixed in `verifiers/v1/mcp/server.py` with regression test `tests/v1/test_mcp_server_latency.py`. Stock vs all Verifiers fixes, 25 tasks x 2 rollouts at c16: setup 4.07 -> 0.12 s, harness 2.50 -> 0.53 s, host time per episode 6.57 -> 0.65 s; the stock figures match production traces (setup 6.3 s, harness 3.2 s median).
- [x] (2026-09-23 08:26Z) Milestone 4: `ngram_gpu` speculative decoding (4 draft tokens) on top of the continuation fix, c16: 77.3 s per collection against 64.9 s (0.84x). Rejected: at temperature 0.8 drafts rarely match and verification costs more than it saves.
- [x] (2026-09-23 08:32Z) LoRA cost diagnostic: the same c16 replay without the rank-4 policy adapter runs 54.5 s per collection against 64.9 s, so policy-LoRA kernels cost 19% of rollout time. Tuning the Punica shrink/expand Triton configs for SM120 (`vllm/lora/ops/triton_ops/README_TUNING.md`) is the next kernel target; it needs another vLLM release and image rebuild, so it was not held for this release.
- [x] (2026-09-23 08:40Z) Release: vLLM `carbonteq-v0.29.1.dev3` (commit `564ff2b43d499f5d17bcee554d126360b768dd98`, source archive SHA-256 `28d20ff20893e1570b3789fb1367af4ca78a8bd71e31c03d43476a6f89aa57d0`, reproducible with `git archive --format=tar.gz --prefix=vllm-0.29.1.dev3/`), pushed to `codex/sm120-attention-platform`; Verifiers `b71ade0a7ac712cdee9e1a4c0e53030d70768aff` pushed to `codex/evaluation-task-selection-v04`. Framework pins, locks and the catalog receipt updated in this repository; runtime images rebuilt with `posttrain-release images publish --framework-version 0.4.4+agentic1`. Pull request carbonteq-ai/posttrain#117.
- [x] (2026-09-23 08:47Z) Eager-mode diagnostic (the VORTEX v1 binding's `enforce_eager: true`) on the continuation-fix stack, c16: 151.9 and 149.8 s per collection against 64.9 s with CUDA graphs, so eager decode costs 2.32x. This is the largest single rollout lever for the VORTEX binding and motivates `c40-4k@2`.
- [x] (2026-09-23 08:52Z) Final c32 replay on the released stack (production c32-4k@2 binding, vLLM dev3): 84.2-84.4 s per collection against the 98.3-98.6 s baseline (1.169x, 1,366 against about 1,170 episodes/h); prefilled tokens 852k -> 191k per collection (-78%); recomputed reusable context 49.4% -> 0.1%; mean TTFT 244 -> 160 ms. The c32 loss that looked like eviction was mostly the same generated-token gap compounding under cache pressure.
- [x] (2026-09-23 13:45Z) Milestone 9 run: `lfm26-vortex-v2-agentic-20260923-r1` (dstack `pt-86530164a4bfd39df2a9d081`, job image `registry.lan/carbonteq/posttrain-lab/posttrain-job@sha256:b56b3377cda8e9fe139e7156711e77f3d82ec127a14b53cee8dbe8be380b55e7`) succeeded at 20/20 updates with model, checkpoint (10, 20), controller-state, summary and recovery artifacts. Against the v1 qualification run `pt-2c092d442e9b2085151e0c33`: wall time 25,450 -> 17,070 s (7.07 -> 4.74 h, 1.49x); mean step 1,248 -> 830 s (1.50x); seconds per generated candidate group 22.0 -> 13.4 (1.64x) while generating 9% more groups (1,236 against 1,136; 3.5 against 2.65 refill rounds per update); rollout throughput 390 -> 633 tokens/s (1.62x). Learning behavior matched: mean reward over updates 0.341 -> 0.350, last-five mean 0.278 -> 0.341, final 0.249 -> 0.393, mean clipped completions 0.268 -> 0.229, entropy 0.173 -> 0.169. Production traces confirm the host path: after the first collection (13 s setup while zygotes boot and uv prepares the harness environment once), episode setup is 0.16-0.27 s and harness time 0.45 s median, against 6.3 s and 3.2 s before.
- [ ] Held-out evaluation (20 tasks x 3, `lfm26_automationbench_heldout_eval.yaml`) of the base model, the v1 adapter and the v2 adapter on the new stack, each with `--model-from-run` for adapters.
- [ ] Milestone 6 first pass: KV cache size sweep (8, 16, 32 GiB) and CPU KV offload (32 and 64 GiB, all tokens; 32 GiB prompt only), queued.
- [ ] Milestone 2: live-episode benchmark on the larger tasks with the real vLLM server (AutomationBench environment now installed; the workstation's `~/venvs/abench-eval` must be reinstalled from a `git archive` of Verifiers, see Surprises).
- [ ] Milestone 3: prefix caching across turns and samples, including hybrid models and batch invariance.
- [ ] Milestone 4: draft-free speculative decoding for tool calls.
- [ ] Milestone 5: prefill and CUDA-graph configuration for long agent contexts.
- [ ] Milestone 6: KV capacity and retention between turns.
- [ ] Milestone 7: serving-path CPU overhead in the token endpoint and renderer.
- [ ] Milestone 8: harness and tool-execution overhead in the AutomationBench environment runtime.
- [ ] Milestone 9: stacked configuration, dashboard section, and recommendation for catalog bindings.

## Surprises & Discoveries

- Observation: two rollouts of the same AutomationBench task almost never start from the same world.
  Evidence: validating the same `initial_state` twice produced different worlds for 797 of 800 tasks, mostly because `google_sheets` row ids, `meta.current_time`, and Salesforce/Gmail/Zoom timestamps and ids are generated by `uuid4()` and `datetime.now()` at model construction (`automationbench/schema/*`). Consequence for this plan: live-episode comparisons must use greedy decoding and compare trajectories structurally (tool names, arguments modulo generated ids, final score), not byte-for-byte, unless world construction is made deterministic.
- Observation: prefix caching in the production rollout path only survives within one collection.
  Evidence: the TRL async session resets the prefix cache on every policy synchronization and sleeps the engine after every collection (`trl/generation/vllm_generation.py:1181-1210`, `trl/generation/async_vllm_session.py:265-281` in the installed TRL fork). That is correct after a weight change, so the benefit available is within a collection: across the turns of one episode and across the samples of one task.
- Observation: the veRL backend hard-codes prefix caching off.
  Evidence: `packages/train/src/posttrain/train/backends/verl/worker.py:208`.
- Observation: in the production LFM2.5 rollout configuration, prefix caching is worth only about 10% because the 4 GiB KV cache cannot hold 32 growing conversations, so cached prefixes are evicted before the next turn.
  Evidence: replay of the 50-episode sample at concurrency 32, three repeats: baseline 1,160-1,173 episodes/h, 37% of prompt tokens served from cache, 1.70-1.72 M tokens prefilled per repeat, mean TTFT 1.13-1.18 s (p95 4.3-4.8 s); with prefix caching off, 1,057-1,060 episodes/h and 2.72 M tokens prefilled. A 6-episode smoke run, where the cache is not under pressure, hit 82%.
- Observation: the token endpoint's JSON validation and prompt echo are negligible per turn.
  Evidence: parsing a 23,000-token request body takes about 1.0 ms and dumping the echoed prompt about 0.6 ms on the development host, against a mean TTFT of 1,180 ms and multi-second turns. Milestone 7's endpoint changes are therefore deprioritized; renderer re-rendering moves to Milestone 2, where live episodes can measure it.
- Observation: draft-free speculative decoding methods in the fork need packages that are not installed, and the CPU methods disable asynchronous scheduling.
  Evidence: `ngram` needs `numba`, `suffix` needs `arctic-inference==0.1.1` (`vllm/config/speculative.py:1619`); both force-disable async scheduling (`vllm/config/vllm.py:1439-1484`). `ngram_gpu` needs neither package and keeps async scheduling.

- Observation: the fork (and upstream vLLM since #52216) retains Mamba/short-conv prefix-cache states only at "replay boundaries" computed from the prompt (`prefix_cache_retention_interval` defaults to 0), so a multi-turn continuation can never reuse the tokens the model generated on the previous turn.
  Evidence: per-turn diagnostic at 8 episodes: lost reusable tokens minus the previous turn's output tokens is 1-16 on every turn (the partial block), i.e. exactly the generated output is re-prefilled. With `prefix_cache_retention_interval: null` (dense) the same run loses 0.1% instead of 10.3% and prefills 34,135 instead of 68,663 tokens. Fix: `get_replay_boundaries` adds the last computed position once a request decodes.
- Observation: at c32 the recomputed context is mostly eviction, not only the generated-output gap.
  Evidence: c32 production loses 49.4% (662,932 tokens) of reusable context per collection against 9.6% at c16, where the loss is almost entirely the generated-output gap.
- Observation: CPU KV offload survives the per-repeat prefix-cache reset, so a replay's later repeats reuse KV from earlier repeats.
  Evidence: best-settings c16 repeats 1-2 reach 84.6-86.8% prefix hit and zero group-duplicate prefill against 80.9% on repeat 0; RL resets caches on each policy synchronization, so only repeat 0 is valid for offload configurations until the replay also resets the offload tier.
- Observation: per-rollout process startup is Python imports, not waiting.
  Evidence: `python -c "import automationbench_v1.tools"` takes 2.2 s (it imports all of `verifiers.v1` and the judge); the harness chat program imports `openai`, `mcp` and `httpx` (0.75 s) and lazily `openai.resources` on its first request (0.44 s). The one-second port-file poll adds about 0.75 s only at low concurrency; at c16 sixteen concurrent cold imports dominate.
- Observation: every MCP tool call waited about 40 ms for a TCP delayed ACK.
  Evidence: in a standalone uvicorn server even a plain ASGI endpoint answered in 41 ms when its listener came from `socket.socket(AF_INET, SOCK_STREAM)` and in 1.0 ms when created with `IPPROTO_TCP` or with `TCP_NODELAY` on the listener; asyncio sets `TCP_NODELAY` per connection only when `sock.proto == IPPROTO_TCP`, which a proto-0 socket does not report. The Verifiers tool server bound its listener that way. In live AutomationBench replays the server handled a call in 3.2 ms while the client waited 46 ms; with about 2.1 tool calls per turn that was about 90 ms per turn.
- Observation: every colocated LFM2.5 VORTEX rollout binding (`…rollout-local-c40-4k@1`) forces eager mode, while the replay baseline and the 0.4.4 c32 profile use CUDA graphs.
  Evidence: `enforce_eager: true` at `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml` (c40-4k@1); the rationale in `docs/plan/trl-mtp-turboquant.md` is graph-pool memory during colocated backward, which is not binding for a 2.6B model at a 0.20 memory fraction. The completed 50-update adaptive-GRPO run used the CUDA-graph c32-4k@2 profile.
- Observation: the September 15 held-out comparison did not evaluate the trained VORTEX adapter.
  Evidence: `eval-lfm26-automationbench-heldout20x3-olmo3-adaptive-oversample10x4-12k-lfmrec-20260915-r1` (`pt-865d2eeb8304e20d46d839b7`) served `models/lfm2.5-2.6b@bf16` and scored a task mean of 0.3917 with 7/20 tasks always solved, identical to both base-model evaluations (r5, r6); its traces name the base model. Held-out claims for that run therefore compare the base model with itself.
- Observation: the previous VORTEX run's rollout phase dominates its wall time.
  Evidence: `pt-2c092d442e9b2085151e0c33` logs: 20 updates in 25,450 s, step time 715-2,154 s (mean about 1,270 s), final mean reward 0.2487, clipped-completion ratio 0.10-0.50.
- Observation: the workstation's `~/venvs/abench-eval` never had a working `verifiers` import.
  Evidence: `~/src/abench-src/verifiers` is a flattened export (the package directory's contents at the root), so the install registered metadata only. Local runs use a `git archive` of the pinned commit instead.

## Decision Log

- Decision: configuration tuning is settled once, early, and then frozen as the "best vLLM settings" reference; the plan's optimizations are code, scheduler, kernel and harness changes measured against that reference.
  Rationale: the user's goal is optimizations beyond what vLLM settings can give. The production binding stays recorded as the "today" baseline, but speedups are attributed against the best settings, so a configuration win is never counted as an optimization. The best-settings candidate is bundle 1 (`max_model_len` 32,768, 16,384 batched tokens, vLLM's default attention backend, 32 GiB CPU KV offload of all tokens, no detokenization); whichever of it and the production binding is faster at c16 becomes the reference. Priority optimizations, both measurable by replay: session-aware KV retention (keep an episode's prefix while it waits on a tool call) and in-flight prefix sharing for GRPO groups (compute a group's shared prompt once when its samples start together).
  Date/Author: 2026-09-23 / user, recorded by Claude.
- Decision: the replay measures RL collections, not a steady stream, and its headline metric is collection seconds. Optimization iterations run at concurrency 16 (8 tasks x 2 recorded samples per collection); concurrency 32 (16 x 2) is measured only for the baseline and the final stacked configuration.
  Rationale: the user pointed out that stream-mode numbers mislead for RL. Production collections (for example `apps/lab/.posttrain/work_packages/lfm26_automationbench_gdpo_episode_50_local_8x4_v1.yaml`, 8 prompt groups x 4 trajectories) start every episode together, wait for the slowest (the trainer's barrier), share the group's prompt at the same moment, reset the prefix cache at each policy synchronization, and sample at temperature 0.8 / top-p 0.95. The recording holds two real samples per task, so collections use 2 samples per group; samples are never duplicated, because duplicates would share whole trajectories and overstate cache reuse. The earlier stream-mode results (baseline 1,160-1,173 episodes/h) remain in `outputs/qualification/agentic-benchmark/` for reference but are superseded as the baseline. Concurrency 16 keeps iteration time manageable, per the user.
  Date/Author: 2026-09-23 / user, recorded by Claude.
- Decision: audit every existing binding setting, not only new optimizations.
  Rationale: the user asked to check whether current configs are wrong. In stream mode the stock-vLLM-defaults reference was 2.1% faster than the production binding (1,191-1,194 vs 1,160-1,173 episodes/h), so at least one binding value costs throughput. The c16 audit flips one setting at a time: attention backend, batched-token budget, KV size, offload, and stock defaults. Separately, `max_model_len` 24,576 truncates long episodes: 2 of 357 recorded calls failed with 25,431-token prompts.
  Date/Author: 2026-09-23 / Claude.
- Decision: the replay baseline serves the base model with a synthetic rank-4 LoRA adapter over the same projections production trains (`make_lora.py`), rounded up to vLLM's supported rank 8 exactly as the framework does.
  Rationale: production rollouts always run with the policy LoRA, whose kernels cost time; the adapter's values do not affect replay because output lengths are fixed, so a synthetic adapter with the right shapes is faithful.
  Date/Author: 2026-09-23 / Claude.
- Decision: the replay engine keeps the binding's `kv_cache_memory_bytes` (4 GiB) but raises `gpu_memory_utilization` from the colocated 0.09 to 0.85.
  Rationale: the replay engine does not share the GPU with a trainer; KV capacity, which is what shapes scheduling, stays pinned to production by `kv_cache_memory_bytes`.
  Date/Author: 2026-09-23 / Claude.
- Decision: the production binding is the best-settings reference at c16.
  Rationale: the best-settings bundle's only valid repeat (67.7 s) was not faster than production (66.4 s); its later repeats are inflated by the offload tier surviving the cache reset. CPU offload is revisited at c32, where eviction dominates, once the replay resets the offload tier between repeats.
  Date/Author: 2026-09-23 / Claude.
- Decision: rollout startup is fixed with an opt-in fork server in the Verifiers subprocess runtime, enabled process-wide by `VF_FORK_SERVER=1` and `VF_FORK_SERVER_PRELOAD`.
  Rationale: tool servers run in their own runtime built from the server's config (`verifiers/v1/mcp/launch.py` `_serve`), so a per-runtime setting in the environment config would not reach them; an environment variable covers every subprocess runtime a worker creates and is what a job image sets. Programs the zygote cannot reproduce exactly (other interpreters, interpreter options, different `PYTHON*` startup variables) fall back to exec.
  Date/Author: 2026-09-23 / Claude.
- Decision: the VORTEX comparison run uses the best vLLM settings (the 0.4.4 CUDA-graph rollout profile scaled to 40 sequences) plus the new software, as `vortex-20-local-v2`.
  Rationale: the user's goal is optimizations measured against the best vLLM settings, and the v1 package predates that profile. Replay results attribute the gain between configuration (eager against CUDA graphs) and software (continuation fix, host path), so one RL run can compare against the v1 run without hiding which part came from where.
  Date/Author: 2026-09-23 / Claude.

- Decision: tool-result caching is out of scope.
  Rationale: the user asked to skip it. The survey also showed it would first require making AutomationBench world construction deterministic per task, which changes environment semantics.
  Date/Author: 2026-09-23 / user, recorded by Claude.
- Decision: the baseline is the current production configuration, frozen before any change: vLLM fork commit `6c092f699e` on branch `codex/sm120-generic-invariant-gemm`, with each model's existing catalog binding values (for LFM2.5 the `lfm2.5-2.6b-vllm-local-c32-4k@2` rollout binding and `eval-local@2` eval binding in `apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml`). Stock vLLM defaults are recorded as a secondary reference only.
  Rationale: the user wants speedups stated against what runs today. Freezing a commit and binding values makes every later number comparable.
  Date/Author: 2026-09-23 / Claude, per user request.
- Decision: each optimization is measured twice: alone against the baseline, and stacked on the optimizations accepted before it.
  Rationale: isolated numbers show what each change buys; stacked numbers show the real total and catch interactions (for example, speculative decoding changing prefix-cache hit rates).
  Date/Author: 2026-09-23 / Claude.
- Decision: the benchmark starts as a qualification script under `scripts/qualification/agentic_benchmark/` in this repository, not as a new `serve.benchmark` workload type.
  Rationale: the frozen baseline (`docs/post-training/README.md`, constraint-driven serving capacity amendment) gives `Workload` ownership of representative prompt populations and `serve.benchmark` ownership of point-level capacity evidence. A multi-turn replay workload would be a real extension of that contract. Proving the measurement first avoids amending the baseline before the method is known to work; promotion into `serve.benchmark` is a follow-up that will need a baseline amendment.
  Date/Author: 2026-09-23 / Claude.
- Decision: this plan does not change the frozen product baseline.
  Rationale: every change is either a binding value (prefix caching, speculative method, scheduler limits), a backend-private implementation detail (token endpoint serialization, renderer bridging, trace I/O), an environment-runtime detail inside the AutomationBench adapter repository, or a vLLM fork change. None changes a public operation, selection meaning, or trace authority.
  Date/Author: 2026-09-23 / Claude.

## Outcomes & Retrospective

Nothing yet.

## Context and Orientation

Terms used in this plan:

An episode is one attempt by the model at one AutomationBench task, from the first prompt to the final answer. A turn is one model call inside an episode. A tool call is a structured request the model emits (a function name plus JSON arguments) that the environment executes. Prefill is the model's pass over the input tokens of a turn; decode is generating its output tokens one step at a time. Time to first token (TTFT) is the delay from sending a turn to receiving its first output token, and is dominated by prefill. The KV cache is the per-token attention state vLLM keeps on the GPU; prefix caching lets vLLM reuse the KV cache of an earlier request whose token prefix matches, so a later turn only prefills the new tokens. Hybrid models (LFM2.5 with short-convolution layers, Qwen3.5 with gated-delta-net layers) also keep a recurrent state; vLLM caches it only at block boundaries in its `align` mode (`mamba_cache_mode`). Speculative decoding proposes several future tokens cheaply and verifies them in one model step; "draft-free" methods propose tokens by copying from the prompt or from previous outputs (n-gram or suffix matching) instead of running a second model. Batch invariance is the `VLLM_BATCH_INVARIANT=1` mode in which a request's output bits do not depend on its batch; it is documented in `docs/architecture/vllm-inference-optimization.md`.

Repositories involved, each checked out as a sibling of this repository under `/home/hammad/projects`:

- `rl` (this repository): framework code, catalog bindings, this plan, and the benchmark scripts.
- `vllm-sm120`: the CarbonTeq vLLM fork, branch `codex/sm120-generic-invariant-gemm`, ledger `CARBONTEQ_FORK.md`. It is synchronized to the workstation with `/home/hammad/projects/k2-horizon-inference/scripts/remote-dev.sh sync` into `~/src/vllm-sm120-dev`.
- `verifiers-environments`: holds the AutomationBench adapter at `environments/automationbench_v1`; the catalog pins commit `a6d779f` (`packages/catalog/src/posttrain/catalog/base/environments.yaml:7`). It currently has uncommitted edits by another contributor to `judge.py`, `episode_prompt.py` and `limited_tools.py` and an untracked `judge_budget.py`; do not include those in any commit made by this plan.
- `automationbench`: the CarbonTeq fork of the benchmark itself (tasks, tools, scoring).

How an AutomationBench episode runs today. Tasks are Python data in `automationbench/domains/<domain>/tasks.py`; there are 800, in sales, marketing, operations, support, finance, hr (100 each) and simple (200). Each task carries an `initial_state` (a simulated world of 47 apps as a Pydantic `WorldState`), the tools it may use, and assertions checked against the final world. Tools are plain Python functions over that world; two of them (`chatgpt` completions and `salesforce_sosl_query`) call OpenAI when `OPENAI_API_KEY` is set. The Verifiers v1 "null" harness runs one episode as separate processes: a chat program and a tool server (`python -m automationbench_v1.tools`) reached over MCP HTTP. For each tool call the tool server fetches the whole episode state over HTTP, validates it, runs the tool, serializes the world, and writes it back (`verifiers` `mcp/server.py:169-247`). Tool calls within one model reply run sequentially. Measured setup cost is a median of 6.3 seconds per rollout, mostly process start and imports.

How rollouts reach vLLM. Every AutomationBench training work package uses TRL's asynchronous colocated mode: Verifiers worker processes render prompts themselves and send exact token ids to an in-process vLLM `AsyncLLM` through a local HTTP endpoint (`packages/train/src/posttrain/train/backends/trl/policy_endpoint.py`, route `/inference/v1/generate`). Evaluations start a managed `vllm serve` process and call its OpenAI-compatible chat API (`packages/serve/src/posttrain/serve/backends/vllm/server.py`). Serving values come from inference bindings; `enable_prefix_caching` defaults to `False` in `packages/serve/src/posttrain/serve/profiles/base.py:116`, is forwarded to TRL only when a binding sets it (`packages/train/src/posttrain/train/backends/trl/common.py:119-124`), and is forced off for veRL (`packages/train/src/posttrain/train/backends/verl/worker.py:208`). The current LFM2.5 AutomationBench rollout binding (`lfm2.5-2.6b-vllm-local-c32-4k@2`) sets prefix caching and CUDA graphs on, `max_model_len` 24576, 32 sequences and 32768 batched tokens; episodes use `max_turns: 12` and 32 concurrent episodes.

Recorded episodes available for replay. `outputs/qualification/lfm26-oversample-training-sample-50-20260920/native-payloads.jsonl` holds 50 native Verifiers traces from LFM2.5-2.6B over 25 tasks in all seven domains, 2 to 12 model calls each, largest prompts a median 11.7k and at most 22.9k tokens. Each trace's `nodes` carry the exact `token_ids` and `mask` of every turn, so turns can be re-sent to vLLM token for token. The full population of 1136 traces lives on the remote Trackio server `trackio.lan`.

Larger tasks for the live benchmark. By number of available tools and assertions, the largest tasks include `support.reamaze_intercom_sync`, `sales.full_sales_cycle_orchestrator`, `support.intercom_sf_opportunity_alerts`, `support.helpcrunch_zoho_desk_bridge`, `operations.zoom_board_meeting`, `support.intercom_usage_health_scoring`, `support.zendesk_freshdesk_sync`, `sales.contract_renewal_coordinator`, `operations.docusign_annual_review`, `support.intercom_demo_scheduling`, `sales.zoom_calendar_conflict` and `sales.cross_platform_account_health_score`. The task data has no expected turn count, so size is measured from the runs themselves.

## Plan of Work

The work is split into milestones that each end in a measured, comparable result. Milestones 1 and 2 build the measuring instrument and freeze the baseline. Milestones 3 to 8 are independent optimizations, each measured alone against the baseline and then stacked. Milestone 9 assembles the stacked configuration and records it.

### Milestone 1: replay benchmark and frozen baseline

Scope: a benchmark that needs no AutomationBench environment. It replays the recorded episodes turn by turn against a vLLM engine, using each turn's recorded prompt token ids and generating exactly the recorded number of output tokens with `ignore_eos` so the decode work matches the recording. Episodes run concurrently at the production concurrency (32). Between turns it waits the recorded tool latency, or zero with `--tool-wait zero`. Because each turn's prompt is the recorded prompt, not built from the model's new output, the workload is identical across configurations, which makes server-side optimizations (prefix caching, prefill limits, KV retention, CUDA graphs) directly comparable.

Create `scripts/qualification/agentic_benchmark/replay.py`. Inputs: `--traces` (a native-payloads JSONL), `--model`, `--binding-config` (a JSON file of vLLM engine arguments, so a configuration is data, not code), `--concurrency`, `--repeats`, `--label`, `--output`. It starts one in-process `AsyncLLM` with the given engine arguments, submits turns through `generate()` with token-id prompts (matching the production token-in path), and records per turn: submit time, first-token time, finish time, prompt tokens, cached prompt tokens (from the request's `num_cached_tokens`), and output tokens. It also snapshots vLLM's Prometheus counters for prefix-cache queries and hits, prompt and generation tokens, and preemptions before and after. The output JSON carries the resolved engine arguments, vLLM commit, GPU name, trace-file digest, and per-episode and aggregate results.

Create `scripts/qualification/agentic_benchmark/summarize.py`, which reads several result files and prints a comparison table against a named baseline file: episodes per hour, mean and p95 episode wall time, mean and p95 TTFT, mean per-turn latency, output tokens per second, prefill tokens computed per episode, prefix-cache hit rate, and speedup versus baseline with a bootstrap 95% interval over repeats.

Freeze the baseline: run the replay with the LFM2.5 rollout binding's engine arguments at fork commit `6c092f699e`, three repeats, and store the result under `outputs/qualification/agentic-benchmark/baseline/` with the binding JSON beside it. Also run the stock-defaults reference once.

Acceptance: `summarize.py` prints the baseline row and three repeats whose episodes-per-hour spread is under 5%. A second run of the same configuration on another day lands within that spread.

### Milestone 2: live-episode benchmark on larger tasks

Scope: run real AutomationBench episodes on the workstation so that host-side costs (tool execution, harness processes, rendering) and model-dependent effects (speculative acceptance, turn counts) are measured. This needs the AutomationBench environment and its Python dependencies installed in a separate virtual environment on the workstation, which requires downloading packages from PyPI; that download needs explicit approval from the user before it happens.

Create `scripts/qualification/agentic_benchmark/live.py`. It serves the model with `vllm serve` using the binding's arguments (the evaluation path), runs a fixed task list through the AutomationBench Verifiers environment with greedy decoding and a fixed seed, and records per episode: turns, tool calls, per-turn TTFT and latency from the server's request metrics, tool execution time and harness overhead from the trace timing, final reward and `task_completed_correctly`. The task list is `scripts/qualification/agentic_benchmark/tasks_large.txt`: the twelve largest tasks named in Context and Orientation plus four medium tasks from finance and hr, with `max_turns` 12 (the production value) and a second profile with `max_turns` 30 for long episodes.

Correctness guard for live runs: because world construction draws ids and times at random (see Surprises & Discoveries), trajectories are compared structurally: the sequence of tool names and arguments with generated ids masked, and the final score per task. With greedy decoding under batch invariance, an optimization that must not change outputs (prefix caching, CUDA graphs, harness changes) must reproduce the baseline's structural trajectories on every task.

Acceptance: the baseline live run completes all tasks in three repeats; per-task scores and structural trajectories repeat exactly across repeats under greedy decoding with invariance on.

### Milestone 3: prefix caching across turns and samples

Scope: make sure every agentic path reuses the KV cache for the shared prefix of consecutive turns and of the samples of one task within a collection, including hybrid models, and measure it. Work items: add explicit `enable_prefix_caching: true` to the AutomationBench rollout and eval bindings that leave it unset, so behavior no longer depends on each vLLM version's default; make `as_cli_args` in `packages/serve/src/posttrain/serve/profiles/base.py` emit `--no-enable-prefix-caching` when a binding sets it false, so the value is always explicit; make veRL's `enable_prefix_caching` follow the binding instead of the hard-coded `False` in `packages/train/src/posttrain/train/backends/verl/worker.py:208`; and validate hybrid prefix caching (`mamba_cache_mode` `align`) under batch invariance for LFM2.5 and Qwen3.5, which the fork currently warns is unvalidated for GDN (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`). The invariance validation is a vLLM-fork task: a prompt that hits a cached prefix must produce the same bits as one computed in full; if a cache hit can resume a GDN prompt off a 64-token boundary, align the cache checkpoint positions to 64 tokens as the scheduler already does for prefill chunks.

Acceptance: replay with prefix caching on shows a cache hit rate close to the share of shared prefix tokens in the recorded episodes and a reduction in prefilled tokens and TTFT; live runs reproduce the baseline trajectories exactly; new unit tests in `packages/serve/tests` cover the explicit CLI flag; the fork's determinism tests gain a prefix-cache-hit bit-equality test for GDN and short-conv.

### Milestone 4: draft-free speculative decoding for tool calls

Scope: tool-call arguments often copy strings from the conversation (record ids, e-mail addresses, names), and JSON structure is predictable, so copying-based proposers can accept several tokens per step. Start with `ngram_gpu`, which needs no new packages and keeps asynchronous scheduling; then, if the download is approved, measure `suffix` decoding (`arctic-inference==0.1.1`) and CPU `ngram` (`numba`). This milestone uses live episodes, because acceptance depends on the model's own outputs.

Correctness guard: with greedy decoding, speculative decoding must not change outputs in exact arithmetic, but it changes batch shapes, so under batch invariance the trajectories must still reproduce exactly; without invariance, task success and reward must stay within the baseline's repeat-to-repeat spread.

Acceptance: a table of acceptance length, decode speedup, and episode speedup per method and per model, with guards reported; a recommendation of which method, if any, to add to catalog bindings, recorded in the Decision Log.

### Milestone 5: prefill and CUDA-graph configuration

Scope: measure time to first token for long agent prompts and tune the knobs that govern it: `max_num_batched_tokens`, `long_prefill_token_threshold`, chunked prefill, and CUDA graphs (several older bindings still run `enforce_eager: true`). Replay is the main instrument. Record whether long prefills stall decode for other episodes by comparing per-turn decode latency with and without concurrent long prefills.

Acceptance: a recommended setting per model with TTFT and episodes-per-hour against baseline; unchanged trajectories in live runs.

### Milestone 6: KV capacity and retention between turns

Scope: while the environment runs a tool, the episode's KV cache is idle and can be evicted under memory pressure, forcing a full prefill on the next turn. Measure eviction from preemption counters and cache-hit shortfall at 32 and 40 concurrent episodes. Evaluate vLLM's CPU KV offload (`--kv-offloading-size`, with `offload_prompt_only` set false so assistant turns are offloaded too) and, separately and only as information for the user, the capacity gain from an FP8 KV cache, which changes numerics and therefore needs a policy decision before any binding uses it.

Acceptance: measured cache-hit rates and episodes per hour with and without offload at the production concurrency; a written recommendation.

### Milestone 7: serving-path CPU overhead

Scope: the production token endpoint does per-turn work the trainer does not need. It validates and echoes the full `prompt_token_ids` of every request as JSON (`packages/train/src/posttrain/train/backends/trl/policy_endpoint.py:281-285, 358-377`), leaves vLLM detokenization on although only token ids are consumed (`policy_endpoint.py:307-324`; vLLM `SamplingParams.detokenize` defaults to true), writes episode JSONL synchronously on the serving event loop (`packages/train/src/posttrain/train/integrations/verifiers.py:997-998, 1151-1182`), and the default renderer re-applies the chat template to every message prefix each turn (`renderers/default.py:119-150`, installed package). Changes: stop echoing prompt ids when the caller does not request them, set `detokenize=False` for token-only requests, move JSONL and trace writes to a background writer, and add incremental bridging for the LFM2.5 and K2 renderers so a turn renders only the new messages. These are backend-private changes in this repository and the renderer package; they must not change token ids.

Correctness guard: token ids sent to vLLM and returned per turn are identical to the baseline for every replayed turn (checked by digest), and existing endpoint tests pass.

Acceptance: host CPU time per turn and episodes per hour against baseline; unchanged token digests.

### Milestone 8: harness and tool-execution overhead

Scope: in the AutomationBench environment runtime, each tool call moves the whole episode state over HTTP twice and re-validates it several times, tool calls in one reply run sequentially, and each rollout spends a median 6.3 seconds starting processes. Measure these from live-run trace timing first. Candidate changes, each kept only if measured: keep the world in the tool server between calls instead of fetching and writing it back per call; skip the write-back when a tool did not change the world (partly done for `search_tools` already); reuse warm tool-server processes across episodes; and run independent read-only tool calls of one reply concurrently. These live in the `verifiers-environments` adapter and possibly the `verifiers` fork, so each change needs a commit in that repository followed by a catalog pin update here, in that order.

Correctness guard: per-task final worlds and scores identical to the baseline under greedy decoding (structural comparison as in Milestone 2).

Acceptance: harness seconds per episode and episodes per hour against baseline.

### Milestone 9: stacked configuration and dashboard

Scope: combine every accepted optimization, run replay and live benchmarks for LFM2.5 first and then Gemma-4-12B, Gemma-4-E4B, K2-Horizon-7B and Qwen3.5-2B, and write results to the dashboard at `https://claude.ai/artifact/96Hxq6YneYEXCsv2EZAb3G` under a new "Agentic workloads" section: one row per configuration with episodes per hour, TTFT, prefill tokens, speedup versus baseline (isolated and stacked) and the guard outcome. Propose binding changes in `apps/lab/.posttrain/catalog/*.yaml` as new binding versions (never editing an existing versioned binding in place).

Acceptance: the dashboard shows the baseline and every configuration with guards; this plan's Outcomes section states the final stacked speedup per model.

## Concrete Steps

All benchmark commands run on the workstation `carbonteq-ai-workstation.lan` with the fork synchronized, from the environment used for earlier inference work:

    source /tmp/remote_env.sh   # sets PYTHONPATH to ~/src/vllm-sm120-dev and CUDA paths
    cd ~/src/rl-agentic          # copy of this repository's scripts/qualification/agentic_benchmark
    python replay.py --traces native-payloads.jsonl --model <snapshot> \
        --binding-config bindings/lfm25-rollout-c32-4k.json --concurrency 32 \
        --repeats 3 --label baseline --output results/baseline.json
    python summarize.py --baseline results/baseline.json results/*.json

The binding JSON files under `scripts/qualification/agentic_benchmark/bindings/` are produced from catalog bindings by `scripts/qualification/agentic_benchmark/export_binding.py`, which resolves a binding through the catalog and writes the vLLM engine arguments it would produce, so the benchmark always runs the configuration the framework would run.

Unit tests for framework changes run from the repository root:

    uv run pytest packages/serve/tests packages/train/tests -q
    uv run ruff check . && uv run pyright && uv run lint-imports

This section will be filled with observed transcripts as milestones complete.

## Validation and Acceptance

The plan is complete when, for LFM2.5-2.6B at the production concurrency, the dashboard shows the frozen baseline, each accepted optimization alone, and the stacked configuration, each with episodes per hour, TTFT and prefilled tokens, a speedup with a 95% interval over at least three repeats, and a passing correctness guard; and when the same table exists for at least Gemma-4-12B and K2-Horizon-7B on the replay benchmark. Every change to this repository passes the validation ladder in `AGENTS.md`; every vLLM-fork change passes the fork's determinism suites listed in `CARBONTEQ_FORK.md`.

## Idempotence and Recovery

Benchmark runs write new files under a label and never overwrite the baseline directory; rerunning a configuration adds a repeat. The baseline is tied to fork commit `6c092f699e`; if that commit must be rebuilt, check it out on a separate worktree rather than moving the development branch. Catalog changes add new binding versions, so any work package can return to the baseline binding by pinning the old version. Environment-repository changes are committed there first and only then pinned here, so a failed change never leaves this repository pointing at an unpublished commit.

## Artifacts and Notes

Survey evidence (2026-09-23): the recorded 50-trace sample has 2 to 12 model calls per episode (sorted counts: 2, 2, 3, 3, 3, 3, 4 ... 12, 12, 12), and model time was about 201 s of about 204 s agent time in those traces, so on the production 12-turn profile the model dominates wall time; tool and harness costs matter more on the long profile and at high concurrency, where they overlap less with GPU work.

## Interfaces and Dependencies

`scripts/qualification/agentic_benchmark/replay.py` exposes `run_replay(traces_path: Path, engine_args: dict, concurrency: int, repeats: int, tool_wait: Literal["recorded", "zero"]) -> ReplayResult`, where `ReplayResult` is a dataclass holding per-turn records (`submitted_s`, `first_token_s`, `finished_s`, `prompt_tokens`, `cached_tokens`, `output_tokens`), per-episode totals, aggregate metrics, the resolved engine arguments and environment facts (vLLM commit, GPU, trace digest). `summarize.py` exposes `compare(baseline: ReplayResult | LiveResult, candidates: list[...]) -> Table`. `live.py` exposes `run_live(task_list: Path, model: str, engine_args: dict, max_turns: int, repeats: int) -> LiveResult` with per-episode structural trajectories, scores and timing. The benchmark uses only vLLM's public `AsyncLLM` and `vllm serve` interfaces and the installed Verifiers AutomationBench environment; it imports no private framework modules, so it can later move into `serve.benchmark` without rewriting its measurement core.

Revision note (2026-09-23): initial version, written after the three surveys and with tool-result caching removed from scope at the user's request.
