template: sampo@4
---
```sql result
select count(*) as collections, max(c.step) as last_collection,
       avg(case when c.k <= 10 then c.reward end) as reward_first10,
       avg(case when c.k > c.n - 10 then c.reward end) as reward_last10,
       avg(case when c.k > c.n - 10 then c.truncation_rate end) as truncation_last10,
       avg(c.collection_seconds) as collection_seconds_mean,
       sum(c.rollout_seconds) / sum(c.collection_seconds) as rollout_share,
       sum(c.rollouts_attempted) as rollouts_attempted,
       sum(c.rollouts_truncated) as rollouts_truncated,
       avg(c.zero_spread_share) as zero_spread_share,
       avg(c.step_reward_share) as step_reward_share,
       avg(c.anchor_group_size) as anchor_group_size
from (select run_id, step, reward, truncation_rate, collection_seconds, rollout_seconds, rollouts_attempted,
             rollouts_truncated, zero_spread_share, step_reward_share, anchor_group_size,
             row_number() over (partition by run_id order by step) as k,
             count(*) over (partition by run_id) as n
      from collections) c
```

```sql optimizer
select count(u.update_seconds) as updates, max(u.step) as last_update,
       avg(case when u.k <= 10 then u.entropy end) as entropy_first10,
       avg(case when u.k > u.n - 10 then u.entropy end) as entropy_last10,
       avg(case when u.k > u.n - 10 then u.kl end) as kl_last10,
       avg(u.update_seconds) as update_seconds_mean
from (select run_id, step, update_seconds, entropy, kl,
             row_number() over (partition by run_id order by step) as k,
             count(*) over (partition by run_id) as n
      from updates) u
```

```sql peak
select max(reward10) as reward_peak10, max_by(from_step, reward10) as peak_from, max_by(to_step, reward10) as peak_to
from (select floor((k - 1) / 10) as bucket, avg(reward) as reward10, min(step) as from_step, max(step) as to_step
      from (select step, reward, row_number() over (partition by run_id order by step) as k from collections) c
      group by floor((k - 1) / 10)) b
```

```sql span
select duration_seconds from runs
```

```sql rewards
select step, reward from collections order by step
```

```sql curve
select step, entropy, kl from updates order by step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Environment | {{run.environment | default "not recorded"}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| KL penalty | {{run.kl_beta | default "none"}} |
| Batch | {{run.prompts_per_collection | default "?"}} prompt groups × {{run.rollouts_per_prompt | default "?"}} rollouts per collection |
| Update budget | {{run.update_budget | default "the whole collection"}} {{run.update_unit | default ""}} per update, {{run.update_epochs | default "1"}} pass over each collection |
| Length | {{run.max_length | default "?"}} tokens per episode, {{run.max_completion_length | default "?"}} per reply, {{run.context_tokens | default "?"}} served |
| LoRA rank | {{run.lora_rank | default "full fine-tune"}} |
| Training / inference | {{run.training_binding | default "not recorded"}} / {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Collections | {{result.collections | default "0"}} (last at update {{result.last_collection | default "—"}}) |
| Updates | {{optimizer.updates | default "0"}} of {{run.max_updates | default "?"}} (last update {{optimizer.last_update | default "—"}}) |
| Reward, first → last 10 collections | {{result.reward_first10 | round 3 | default "—"}} → {{result.reward_last10 | round 3 | default "—"}} |
| Best 10 collections | {{peak.reward_peak10 | round 3 | default "—"}} (collections at updates {{peak.peak_from | default "—"}}–{{peak.peak_to | default "—"}}) |
| Entropy, first → last 10 updates | {{optimizer.entropy_first10 | round 3 | default "—"}} → {{optimizer.entropy_last10 | round 3 | default "—"}} |
| KL to reference, last 10 updates | {{optimizer.kl_last10 | sci 2 | default "not measured"}} |
| Turns scored with step rewards | {{result.step_reward_share | percent | default "—"}} |
| Mean anchor group size | {{result.anchor_group_size | round 2 | default "—"}} |
| Mean update time | {{optimizer.update_seconds_mean | duration | default "—"}} |
| Mean collection time | {{result.collection_seconds_mean | duration | default "—"}} |
| Collection time spent in rollouts | {{result.rollout_share | percent | default "—"}} |
| Rollouts | {{result.rollouts_attempted | default "—"}} generated, {{result.rollouts_truncated | default "—"}} truncated |
| Truncated, last 10 collections | {{result.truncation_last10 | percent | default "—"}} |
| Error | {{run.error | default "none"}} |
| Error message | {{run.error_message | default "—"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: rewards
x: step
y: [reward]
title: Reward by collection (at its first update)
```

```chart
data: curve
x: step
y: [entropy]
title: Entropy by update
```
