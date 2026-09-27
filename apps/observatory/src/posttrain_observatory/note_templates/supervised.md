template: supervised@2
---
```sql result
select count(update_seconds) as updates,
       min_by(loss, CASE WHEN loss IS NOT NULL THEN step END) AS loss_first, max_by(loss, CASE WHEN loss IS NOT NULL THEN step END) AS loss_last,
       max_by(validation_loss, CASE WHEN validation_loss IS NOT NULL THEN step END) AS validation_loss_last,
       max_by(token_accuracy, CASE WHEN token_accuracy IS NOT NULL THEN step END) AS token_accuracy_last,
       avg(tokens_per_second) as tokens_per_second
from updates
```

```sql span
select duration_seconds from runs
```

```sql curve
select step, loss, validation_loss from updates order by step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| Length | {{run.max_length | default "?"}} tokens |
| LoRA rank | {{run.lora_rank | default "full fine-tune"}} |
| Training binding | {{run.training_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Updates | {{result.updates | default "0"}} of {{run.max_updates | default "?"}} |
| Loss | {{result.loss_first | round 4 | default "—"}} → {{result.loss_last | round 4 | default "—"}} |
| Validation loss (last) | {{result.validation_loss_last | round 4 | default "not measured"}} |
| Token accuracy (last) | {{result.token_accuracy_last | percent | default "not measured"}} |
| Throughput | {{result.tokens_per_second | round 0 | default "—"}} tokens/s |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: step
y: [loss, validation_loss]
title: Loss by update
```
