template: evaluation@1
---
```data result
measures: [eval_rollouts_attempted, eval_rollouts_complete, eval_rollouts_failed, eval_rollouts_truncated, eval_context_overflow_rollouts, duration_seconds]
```

```data tasks
measures: [task_reward, success_rate, valid_repetitions, execution_failures, eval_truncations]
by: [eval_task.task]
order_by: [-task_reward]
```

```data overall
measures: [task_reward:mean, success_rate:mean, valid_repetitions:sum]
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
| Mean task reward | {{overall.task_reward_mean | round 3 | default "—"}} |
| Mean success rate | {{overall.success_rate_mean | percent | default "—"}} |
| Error | {{run.error | default "none"}} |

```table
data: tasks
title: Tasks by reward
```
