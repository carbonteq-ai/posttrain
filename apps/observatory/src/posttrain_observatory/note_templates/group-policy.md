template: group-policy@1
---
```data result
measures: [update_seconds:count, reward:first, reward:last, entropy:first, entropy:last, kl:last, update_seconds:mean, rollout_share, rollouts_attempted:sum, rollouts_truncated:sum, zero_spread_share:mean]
```

```data span
measures: [duration_seconds]
```

```data curve
measures: [reward, entropy, kl]
by: update.step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Environment | {{run.environment | default "not recorded"}} |
| Algorithm | {{run.algorithm | default "not recorded"}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| KL penalty | {{run.kl_beta | default "none"}} |
| Batch | {{run.prompts_per_update | default "?"}} prompt groups × {{run.rollouts_per_prompt | default "?"}} rollouts |
| Length | {{run.max_length | default "?"}} tokens per episode, {{run.max_completion_length | default "?"}} per reply, {{run.context_tokens | default "?"}} served |
| LoRA rank | {{run.lora_rank | default "full fine-tune"}} |
| Training / inference | {{run.training_binding | default "not recorded"}} / {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Updates | {{result.update_seconds_count | default "0"}} of {{run.max_updates | default "?"}} |
| Reward | {{result.reward_first | round 3 | default "—"}} → {{result.reward_last | round 3 | default "—"}} |
| Entropy | {{result.entropy_first | round 3 | default "—"}} → {{result.entropy_last | round 3 | default "—"}} |
| KL to reference (last) | {{result.kl_last | sci 2 | default "not measured"}} |
| Mean update time | {{result.update_seconds_mean | duration | default "—"}} |
| Update time spent in rollouts | {{result.rollout_share | percent | default "—"}} |
| Rollouts | {{result.rollouts_attempted_sum | default "—"}} generated, {{result.rollouts_truncated_sum | default "—"}} truncated |
| Groups with no reward spread | {{result.zero_spread_share_mean | percent | default "—"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: update.step
y: [reward, entropy]
title: Reward and entropy by update
```
