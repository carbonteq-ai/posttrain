template: sampo@3
---
```sql result
select count(u.update_seconds) as updates, max(u.step) as last_update,
       avg(case when u.step <= s.first_step + 9 then u.reward end) as reward_first10,
       avg(case when u.step >= s.last_step - 9 then u.reward end) as reward_last10,
       avg(case when u.step <= s.first_step + 9 then u.entropy end) as entropy_first10,
       avg(case when u.step >= s.last_step - 9 then u.entropy end) as entropy_last10,
       avg(case when u.step >= s.last_step - 9 then u.kl end) as kl_last10,
       avg(case when u.step >= s.last_step - 9 then u.truncation_rate end) as truncation_last10,
       avg(u.update_seconds) as update_seconds_mean,
       sum(u.rollout_seconds) / sum(u.update_seconds) as rollout_share,
       sum(u.rollouts_attempted) as rollouts_attempted,
       sum(u.rollouts_truncated) as rollouts_truncated,
       avg(u.step_reward_share) as step_reward_share,
       avg(u.anchor_group_size) as anchor_group_size
from updates u
join (select run_id, min(step) as first_step, max(step) as last_step from updates group by run_id) s on s.run_id = u.run_id
```

```sql peak
select max(reward10) as reward_peak10, max_by(from_update, reward10) as peak_from, max_by(to_update, reward10) as peak_to
from (select floor((step - 1) / 10) as bucket, avg(reward) as reward10, min(step) as from_update, max(step) as to_update
      from updates group by floor((step - 1) / 10)) b
```

```sql span
select duration_seconds from runs
```

```sql curve
select step, reward, entropy, step_reward_share from updates order by step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Environment | {{run.environment | default "not recorded"}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| KL penalty | {{run.kl_beta | default "none"}} |
| Batch | {{run.prompts_per_update | default "?"}} prompt groups × {{run.rollouts_per_prompt | default "?"}} rollouts |
| Length | {{run.max_length | default "?"}} tokens per episode, {{run.max_completion_length | default "?"}} per reply, {{run.context_tokens | default "?"}} served |
| LoRA rank | {{run.lora_rank | default "full fine-tune"}} |
| Training / inference | {{run.training_binding | default "not recorded"}} / {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Updates | {{result.updates | default "0"}} of {{run.max_updates | default "?"}} (last update {{result.last_update | default "—"}}) |
| Reward, first → last 10 updates | {{result.reward_first10 | round 3 | default "—"}} → {{result.reward_last10 | round 3 | default "—"}} |
| Best 10 updates | {{peak.reward_peak10 | round 3 | default "—"}} (updates {{peak.peak_from | default "—"}}–{{peak.peak_to | default "—"}}) |
| Entropy, first → last 10 updates | {{result.entropy_first10 | round 3 | default "—"}} → {{result.entropy_last10 | round 3 | default "—"}} |
| KL to reference, last 10 updates | {{result.kl_last10 | sci 2 | default "not measured"}} |
| Turns scored with step rewards | {{result.step_reward_share | percent | default "—"}} |
| Mean anchor group size | {{result.anchor_group_size | round 2 | default "—"}} |
| Mean update time | {{result.update_seconds_mean | duration | default "—"}} |
| Update time spent in rollouts | {{result.rollout_share | percent | default "—"}} |
| Rollouts | {{result.rollouts_attempted | default "—"}} generated, {{result.rollouts_truncated | default "—"}} truncated |
| Truncated, last 10 updates | {{result.truncation_last10 | percent | default "—"}} |
| Error | {{run.error | default "none"}} |
| Error message | {{run.error_message | default "—"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: step
y: [reward, entropy]
title: Reward and entropy by update
```
