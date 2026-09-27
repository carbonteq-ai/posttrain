template: preference@2
---
```sql result
select count(update_seconds) as updates,
       min_by(loss, CASE WHEN loss IS NOT NULL THEN step END) AS loss_first, max_by(loss, CASE WHEN loss IS NOT NULL THEN step END) AS loss_last,
       max_by(preference_accuracy, CASE WHEN preference_accuracy IS NOT NULL THEN step END) AS preference_accuracy_last,
       max_by(reward_margin, CASE WHEN reward_margin IS NOT NULL THEN step END) AS reward_margin_last
from updates
```

```sql span
select duration_seconds from runs
```

```sql curve
select step, loss, preference_accuracy from updates order by step
```

**{{run.job_kind}}** · {{run.status}} · {{span.duration_seconds | duration}} · work package {{run.work_package}}

| Setup | |
| --- | --- |
| Model | {{run.model | default "not recorded"}} |
| Started from | {{run.parent_run | default "base weights"}} {{run.parent_step | default ""}} |
| Learning rate | {{run.learning_rate | sci 1 | default "not recorded"}} |
| KL penalty (beta) | {{run.kl_beta | default "not recorded"}} |
| LoRA rank | {{run.lora_rank | default "full fine-tune"}} |
| Training binding | {{run.training_binding | default "not recorded"}} |

| Result | |
| --- | --- |
| Updates | {{result.updates | default "0"}} of {{run.max_updates | default "?"}} |
| Loss | {{result.loss_first | round 4 | default "—"}} → {{result.loss_last | round 4 | default "—"}} |
| Preference accuracy (last) | {{result.preference_accuracy_last | percent | default "—"}} |
| Reward margin (last) | {{result.reward_margin_last | round 3 | default "—"}} |
| Error | {{run.error | default "none"}} |
| Failed in | {{run.failed_phase | default "—"}} (update {{run.failed_step | default "—"}}) |

```chart
data: curve
x: step
y: [loss, preference_accuracy]
title: Loss and preference accuracy by update
```
