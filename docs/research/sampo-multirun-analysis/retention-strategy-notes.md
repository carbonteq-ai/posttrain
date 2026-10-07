# Independent sampling and retention strategy review

Reviewed the actual controller, recorded curriculum replay, recorded reward sweep, training task windows, and simulation report builder on 2026-10-02. This review concerns offline candidate selection. No training or model update was run.

The smallest useful candidate is a correctly applied local policy loss with independently defined goal, violation, and admission evidence. Keep the current uncertainty lane while qualifying credit. The allocation replay supports neither replacing VORTEX with a yield-only allocator nor reserving an arbitrary quarter of training for practice.

## What the recorded allocator simulations support

The later outcome banks show increased useful-group yield and decreased task coverage when uncertainty weight becomes zero in all seven arms. Complete non-Simple episode share improves in four arms and falls in three. Increasing exploration to 50% also has inconsistent effects on recorded non-Simple completion. These are allocation differences over fixed outcome banks, not learning outcomes.

For FP16 base100, the current setting produces useful-group share 0.934 and coverage 108.0 of 114 common-support tasks; removing uncertainty produces 0.962 and 101.3. Recorded non-Simple completion falls from 0.037 to 0.036. Maximizing usable gradients can give a better utilization number while neglecting conditional procedures.

The replay has no refills and uses 24 candidate groups per round. Production fills an optimizer target through bounded refills, then cuts excess eligible groups. Consequently these results are not equivalent comparisons at 24 optimizer-used groups, equal generated tokens, or equal wall time. Different training arms also have different sibling counts. Full-group banks were selected historically by earlier policies and exclude groups with any truncated sibling. Task outcomes absent from the bank remain unknown. Seeds here quantify allocation variability conditional on that bank, not uncertainty across training seeds.

## A concrete integration failure to address before guard shaping

`AdaptiveCurriculumRuntime.observe_rewards` passes the weighted algorithm reward into the controller. `AdaptiveCurriculumController.observe` rejects a group if any finite reward is outside [0, 1]. Its exploitation score then treats accepted reward sums as fractional successes in a beta model. Both range and meaning matter.

I independently applied the selected terminal cost 1.125 to the same complete, nontruncated hard groups using the existing `groups_for` and `arrays` helpers. These are the shares containing an out-of-range value, which the current controller would reject if supplied unchanged:

| Historical bank | Native reward | Native reward minus guard cost |
| --- | ---: | ---: |
| Original150 | 0.0% | 71.6% |
| Continuation120 | 14.9% | 76.9% |
| Fixed-tools continuation | 17.4% | 80.4% |
| FP16 base100 | 13.9% | 81.8% |
| FP16 higher-LR60 | 11.1% | 79.7% |
| H10050 | 12.7% | 80.3% |
| FP16 continuation100 | 10.0% | 77.3% |

This is a current-controller counterfactual on historical observations. It does not prove which observations historical deployments rejected. The native out-of-range figures also show why curriculum replay must report invalid observations, rather than only sampled outcomes and yield. Replay outcomes can still be counted even when the controller receives no learning evidence from them.

An affine transform can make finite values fit the interval, but an affine safety score is not automatically the controller's assumed probability of successful task progress. Clipping would additionally destroy distinctions and can change usable-group admission. Qualify a structured controller observation: bounded goal progress or full-success evidence for its success model, a separate guard-violation measure, and a useful-credit indicator computed from the exact training credit/admission contract. The current controller API does not yet express that separation. A reward sweep should not be presented as ready to plug into the current runtime.

## Replace an arbitrary practice quota with a testable recheck policy

A generated complete solution is evidence of capability support. It is not evidence that its gradient was optimizer-used, or that the task was mastered. Record the path from native group to admission to optimizer selection, plus the successful branch's actual combined credit. Trigger any consolidation experiment from a verified complete branch that was optimizer-used.

For a candidate practice arm, enqueue that task for fresh collection in subsequent populations. Bound this queue to a declared number of candidate selections, preserve task uniqueness across initial collection and refills, and keep the ordinary zero-variance filter. Enqueue deadlines and queue spillover must be logged. A small reserve, for example two of 24 initial candidate selections, is an experimental budget setting rather than an inferred optimum. Compare it with ordinary VORTEX and equal extra fresh sampling; do not assume it beats a 25% reserve or any other schedule.

The queue should expire after a fixed number of rechecks or consistent independent success evidence. Do not keep selecting a task forever because it once succeeded. Use full procedure success and guard preservation to assess consolidation, not increasing partial score. A group that is all correct has no episode-relative learning signal; losing admission in that case is expected. If maintaining such behavior needs a stored verified solution, call that a demonstration or distillation intervention and qualify its separate objective. Replaying old successful tokens as if freshly on-policy is invalid. Declared repeated optimization of one retained population is another distinct method requiring its own ratios, update schedule, and drift checks.

## Separate unstable learning from sampling noise

The higher-LR advertising example has one complete success among 72 recorded samples, while five siblings already fail at the same step as the success. This directly weakens the claim that a skill had become reliable and then disappeared. If a stationary success probability were 1/72, seeing zero successes in a later set of 30 would have probability about 66%. That is an illustration, not a fitted learning model: checkpoints, task selection, and truncation change over time.

Use frozen checkpoints and a fixed task/world panel to measure full success, prohibited actions, actual procedure retrieval, and final combined action credit. Compare the same checkpoints on repeated independent seeds; match context and sampling parameters. Twelve attempts per task cannot certify rare-branch absence: zero of 12 still permits a success probability around 22% under a one-sided exact 95% bound. At a genuine probability of 1/72, about 215 independent attempts would be needed for a 95% chance of observing at least one success. The measured phenomenon is rare enough that small panels need confidence intervals and explicit inconclusive outcomes.

Score the exact successful branch under later actors using retained conditioning views, alongside fresh execution. Falling branch probability is evidence of actor drift, while fresh execution can fail because of another upstream decision. Teacher-forced branch scores do not prove task execution, and execution averages alone do not locate the changed decision. These measurements together can falsify a retention explanation.

## Minimal qualification sequence

1. Verify current deployed pin and token-local derivative with actual native masks, then keep reward, VORTEX settings, precision, rank, and denominators fixed for a loss-controlled comparison. Keep local centering and turn weight unchanged while making this comparison.
2. Qualify goal/violation/controller evidence separation and guard transitions. Compare the selected terminal-cost candidate with the corrected-loss baseline at matched declared credit scale. Log post-combination local signs and actor drift; a terminal cost can disappear when every sibling violates the same guard.
3. If complete branches are sampled and optimizer-used but remain unreliable, compare bounded event-triggered fresh rechecks with unchanged VORTEX under equal candidate, optimizer-use, and generated-token accounting. Do not combine this with a denominator or LoRA-rank change.
4. If actual procedure retrieval remains the barrier, use exact information availability as a diagnostic and evaluate verified demonstrations as a separately named method. If learned branch probabilities truly fall across frozen checkpoints, then test update size, reference policy, or interference controls.

The first two steps can fail before any sampling intervention is appropriate. The best current candidate is therefore a bounded, auditable experiment sequence, not a claim that recorded-outcome optimization has discovered a good learning system.
