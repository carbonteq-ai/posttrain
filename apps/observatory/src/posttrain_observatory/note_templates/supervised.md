template: supervised@1
---
```data result
measures: [update_seconds:count, loss:first, loss:last, validation_loss:last, token_accuracy:last, tokens_per_second:mean]
```

```data span
measures: [duration_seconds]
```

```data curve
measures: [loss, validation_loss]
by: update.step
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
| Updates | {{result.update_seconds_count | default "0"}} of {{run.max_updates | default "?"}} |
| Loss | {{result.loss_first | round 4 | default "—"}} → {{result.loss_last | round 4 | default "—"}} |
| Validation loss (last) | {{result.validation_loss_last | round 4 | default "not measured"}} |
| Token accuracy (last) | {{result.token_accuracy_last | percent | default "not measured"}} |
| Throughput | {{result.tokens_per_second_mean | round 0 | default "—"}} tokens/s |
| Error | {{run.error | default "none"}} |

```chart
data: curve
x: update.step
y: [loss, validation_loss]
title: Loss by update
```
