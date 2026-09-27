template: distill@2
---
```sql result
select count(update_seconds) as updates,
       min_by(distill_loss, CASE WHEN distill_loss IS NOT NULL THEN step END) AS distill_loss_first, max_by(distill_loss, CASE WHEN distill_loss IS NOT NULL THEN step END) AS distill_loss_last,
       max_by(reverse_kl, CASE WHEN reverse_kl IS NOT NULL THEN step END) AS reverse_kl_last,
       sum(teacher_failures) as teacher_failures
from updates
```

```sql span
select duration_seconds from runs
```

```sql curve
select step, distill_loss, reverse_kl from updates order by step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Student | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Environment | {{run.environment | default "not recorded"}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| Training / inference | {{run.training_binding | default "not recorded"}} / {{run.inference_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Updates | {{result.updates | default "0"}} of {{run.max_updates | default "?"}} |
| Distillation loss | {{result.distill_loss_first | round 4 | default "—"}} → {{result.distill_loss_last | round 4 | default "—"}} |
| Reverse KL (last) | {{result.reverse_kl_last | round 4 | default "—"}} |
| Teacher failures | {{result.teacher_failures | default "0"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: step
y: [distill_loss, reverse_kl]
title: Distillation loss by update
```
