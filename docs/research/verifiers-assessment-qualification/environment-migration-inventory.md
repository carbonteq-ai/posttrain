# Six-environment migration inventory

Read-only source inspection on 2026-10-03; no migration, dependency adoption or publication is claimed.

The external repository is `/home/hammad/projects/verifiers-environments`, branch `codex/evaluation-task-selection`, commit `a6d779fc1fdfde23f86e297125b3381b140cec2f`. Its root `AGENTS.md` is the package authority. Each environment is a standalone project; no root runtime/workspace, sibling imports or Posttrain dependencies belong there. All six built wheels must install together before composed publication.

| Package | Existing scalar behavior | Native-assessment seam and qualification |
| --- | --- | --- |
| `automationbench_v1` 0.4.0 | `AutomationBenchTask.partial_credit`, assertion completion/count metrics | `src/automationbench_v1/taskset.py` snapshot/finalize, `scoring.py`, `tests/test_environment.py` assertion registry and legacy snapshot parity. Preserve existing dirty judge/prompt/budget/tool/taskset work separately. |
| `gsm8k_v1` 0.2.0 | Runtime-verifier `correct`, gold `validate` | `src/gsm8k_v1/taskset.py`; `tests/test_contract.py::test_reward_and_gold_validation_use_runtime_verifier`. Preserve subprocess failure semantics and verifier-result provenance. |
| `mmlu_pro_v1` 0.1.0 | `answer_correct`, parse-success metric | `_parsed_answer` and parser contract fixtures. Emit outcome/parse findings without inventing reasoning credit. |
| `ifeval_v1` 0.1.0 | Strict prompt accuracy plus strict/loose instruction/prompt metrics | `_scores` retains deterministic per-instruction outcomes; registry coverage and strict/loose parity tests. Preserve aggregate scoring. |
| `reasoning_gym_v1` 0.1.0 | Native generator score or selected boxed exactness | `_score`; oracle, syllogism, boxed, deterministic-bank and disjoint-seed fixtures. Preserve generator-specific meaning. Existing README does not qualify a model baseline/catalog release. |
| `math_python_v1` 0.1.0 | Symbolic math correctness and parse metrics | `_verification`; contract fixtures plus bounded Python child success/error/timeout/exit/isolation. Tool lifecycle does not establish mathematical progress; local children do not qualify published images/provider sandboxes. |

All six manifests and locks select native Verifiers `84ab782391bbfe1ac4f4ca32fa612e56d01b5b81`. `scripts/check_boundaries.py` still expects `1f6793f7d46e8a650a54b2a585193b4010578fa6`. Posttrain's data/eval/train manifests and lock select `e6a3d9bbfe6959b97878f451fc721793a232cd5f`. Reconcile these after candidate qualification/publication; do not call a stale boundary constant compatibility evidence.

Baseline commands, with working directory `/home/hammad/projects/verifiers-environments/environments/<package>`:

```bash
uv lock --check
uv sync --locked --python 3.12
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
uv run --frozen pytest -m "not network"
uv run --frozen pytest -m network
uv build --wheel
```

Dataset/source-count tests marked `network` are separate release checks, not implicitly satisfied by deterministic fixtures. From `/home/hammad/projects/verifiers-environments`, the boundary command is `uv run --python 3.12 python scripts/check_boundaries.py`. Until pins are reconciled, record its exact mismatch rather than modifying the expected commit to make it green.

The authoritative gate order remains [the runbook](../../plan/verifiers-assessment-api-and-environment-migration-runbook.md). Assessment API qualification precedes accepting migration, and immutable pin adoption follows published fork commits.
