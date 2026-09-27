template: evaluation@2
---
```sql result
select eval_rollouts_attempted, eval_rollouts_complete, eval_rollouts_failed, eval_rollouts_truncated,
       eval_context_overflow_rollouts, duration_seconds
from runs
```

```sql tasks
select task,
       avg(case when not truncated and not failed then rollout_reward end) as reward,
       sum(case when not truncated and not failed then 1 else 0 end) as valid,
       sum(case when truncated then 1 else 0 end) as truncated,
       sum(case when failed then 1 else 0 end) as failed
from rollouts
group by task
order by reward desc
```

```sql overall
select avg(case when not truncated and not failed then rollout_reward end) as reward_valid,
       avg(case when failed then null else coalesce(rollout_reward, 0) end) as reward_all,
       count(distinct task) as tasks
from rollouts
```

**{{run.job_kind}}** · {{run.status}} · {{result.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Checkpoint | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Environment | {{run.environment | default "not recorded"}} |
| Inference | {{run.inference_binding | default "not recorded"}} |
| Context | {{run.context_tokens | default "?"}} tokens served, {{run.max_completion_length | default "?"}} per reply |

| Result | |
| --- | --- |
| Rollouts | {{result.eval_rollouts_attempted | default "—"}} attempted, {{result.eval_rollouts_complete | default "—"}} complete |
| Failed | {{result.eval_rollouts_failed | default "—"}} |
| Truncated | {{result.eval_rollouts_truncated | default "—"}} (context overflow {{result.eval_context_overflow_rollouts | default "—"}}) |
| Tasks | {{overall.tasks | default "—"}} |
| Mean reward, valid rollouts | {{overall.reward_valid | round 3 | default "—"}} |
| Mean reward, truncated counted | {{overall.reward_all | round 3 | default "—"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```table
data: tasks
title: Tasks by reward
```
