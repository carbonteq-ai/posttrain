# Techniques

Post-training methods. Each technique folder holds overview, **recipes/** (playbooks), and **heuristics.md** (rules of thumb / failure modes).

| Technique | Status |
| --- | --- |
| [sft/](./sft/) | Qwen3.5 QLoRA smoke verified |
| [dpo/](./dpo/) | Qwen3.5 trace-derived preference smoke verified |
| [grpo/](./grpo/) | Qwen3.5 colocated-vLLM + Verifiers smoke verified |
| [inference/](./inference/) | vLLM foundation screening implemented |
| [_template/](./_template/) | Copy for a new method |

Technique code remains in the reusable `posttrain.train` package. These pages
record measured recipes and failure modes; jobs supply task-specific data,
environments, parent artifacts, and acceptance policy.
