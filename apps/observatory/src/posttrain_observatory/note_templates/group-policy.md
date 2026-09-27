template: group-policy@2
---
```sql result
select count(update_seconds) as updates,
       min_by(reward, CASE WHEN reward IS NOT NULL THEN step END) AS reward_first, max_by(reward, CASE WHEN reward IS NOT NULL THEN step END) AS reward_last,
       min_by(entropy, CASE WHEN entropy IS NOT NULL THEN step END) AS entropy_first, max_by(entropy, CASE WHEN entropy IS NOT NULL THEN step END) AS entropy_last,
       max_by(kl, CASE WHEN kl IS NOT NULL THEN step END) AS kl_last,
       avg(update_seconds) as update_seconds_mean,
       sum(rollout_seconds) / sum(update_seconds) as rollout_share,
       sum(rollouts_attempted) as rollouts_attempted,
       sum(rollouts_truncated) as rollouts_truncated,
       avg(zero_spread_share) as zero_spread_share
from updates
```

```sql span
select duration_seconds from runs
```

```sql curve
select step, reward, entropy, kl from updates order by step
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
| Updates | {{result.updates | default "0"}} of {{run.max_updates | default "?"}} |
| Reward | {{result.reward_first | round 3 | default "—"}} → {{result.reward_last | round 3 | default "—"}} |
| Entropy | {{result.entropy_first | round 3 | default "—"}} → {{result.entropy_last | round 3 | default "—"}} |
| KL to reference (last) | {{result.kl_last | sci 2 | default "not measured"}} |
| Mean update time | {{result.update_seconds_mean | duration | default "—"}} |
| Update time spent in rollouts | {{result.rollout_share | percent | default "—"}} |
| Rollouts | {{result.rollouts_attempted | default "—"}} generated, {{result.rollouts_truncated | default "—"}} truncated |
| Groups with no reward spread | {{result.zero_spread_share | percent | default "—"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: step
y: [reward, entropy]
title: Reward and entropy by update
```
