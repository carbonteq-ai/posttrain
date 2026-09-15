# Task and class discovery simulation

This living plan follows `docs/templates/PLAN.md`.

## Purpose / Big Picture

Readers of the v2 curriculum proposal should see individual tasks leave an inventory, produce grouped outcomes, and acquire sampling histories. Class forecasts direct discovery while familiar tasks receive adaptive practice or reassessment. This is a local explanatory simulation, not a trainer implementation or evidence of LLM gains. The frozen product baseline is unchanged.

## Progress

- [x] (2026-09-12) Read the proposal, renderer, and plan template; preserve the original proposal.
- [x] (2026-09-12) Implement deterministic simulation with separate hidden learning behavior and observed controller evidence.
- [x] (2026-09-12) Add compact inventory, selection animation, six scenario controls, history, and task inspection.
- [x] (2026-09-12) Embed in v2, update its Markdown assumptions, and validate behavior and DOM interaction.
- [x] (2026-09-12) Add ordinary learning as the default, with separate starting-ability, learning-speed, and contrast-response presets.
- [x] (2026-09-12) Revalidate 36 scenario/mode/seed runs and the standalone and embedded controls.
- [x] (2026-09-12) Replace per-task presentation events with complete discovery, practice, and recheck lanes; separate success heatmap color from variance length.
- [x] (2026-09-12) Validate lane percentages, unchanged selections, and independent heatmap scales in both generated readers.
- [x] (2026-09-12) Swap the encodings following reader feedback: variance controls tile color; observed reward controls animated bar length.

## Surprises & Discoveries

The existing player has class-level learning curves but no individual-task inventory. Its results cannot establish task discovery or uniqueness across refill rounds. The previous browser preview of a local file was blocked by browser security policy; do not bypass that restriction. Use static and DOM-based checks if the browser remains unavailable.

## Decision Log

On 2026-09-12, use browser-native HTML/CSS/JavaScript with no runtime network dependencies. Keep the hidden per-task success probability outside the controller input. Offer several assumed learning behaviors, and label the mathematical learner as synthetic. Preserve v1; embed the new player only in v2 and publish a standalone copy.

On 2026-09-12, foreground ordinary learning after reader feedback that the initial menu emphasized edge cases. Keep initial task probabilities identical between the ability and signal-response presets so the contrast-dependent learning law can be compared without changing initial conditions. Move hidden-probability inspection into the main controls.

On 2026-09-12, use the supplied screenshots to revise the display around three selection lanes. Keep the engine's sequential conditional draws unchanged; collapse consecutive Select frames only in the presentation. Lane shares describe actual candidate counts, not new policy quotas. Encode observed success with a continuous orange-to-green tile color and within-group variance with a separate purple bar. Use outlines for current-round selection, dotted borders for earlier step selections, and grey/dashes for unknown evidence.

The later reader revision supersedes that color assignment: normalized variance uses pale-to-purple tile color; binary mean reward/success uses green bar length. Animate from the preceding observed value during forward playback. Preserve decreases and missing evidence instead of forcing monotonic improvement or treating unknown values as zero.

The subsequent palette correction restores orange → neutral → green for variance while retaining reward bar length. This changes only the colors and their explanatory labels.

The default learner now starts with heterogeneous task probabilities (10%–88%) and different class means, keeping the common learning rate at 0.14. Class ranges are synthetic assumptions, not inherent category difficulty. Cold-start observed statistics remain unknown and forecasts retain the shared prior; hidden abilities must not leak into selection.

## Outcomes & Retrospective

Added aligned timelines inside the existing player: stacked actual candidate allocation and four separate observed-success histories. They derive only from replay events already reached, include refill candidates, and share the existing controls. Success is recent sampled evidence, not an unbiased qualification metric.

Completed a standalone and embedded player with 64 task identities, separate class forecasts and task histories, four ordinary learning presets and five stress cases, vanilla and bounded active-retention modes, and seeded event replay. Engine checks passed across 36 scenario/mode/seed combinations; standalone and embedded DOM checks passed. This is a documentation simulation, with no GPU training or production controller changes. Visual browser preview remains unavailable because the earlier local-file action was blocked; no workaround was attempted.

The visual revision now reveals a complete candidate request in discovery, practice, and recheck lanes before generation. Each lane displays its realized count and percentage. The inventory separates variance color, observed reward length, and selection outlines. DOM checks verify startup's 100% actual discovery against its 20% reserve, identical selected task identities after presentation grouping, refill exclusions, and independent success/variance encodings. The supplied screenshots grounded the changes, but the revised rendering has not been visually previewed.

## Context and Orientation

The editable proposal is `docs/research/proposals/configurable-curriculum-and-data-preparation-v2.md`. Its renderer is `docs/research/proposals/render_curriculum_v2.py`. Add simulation sources under `docs/research/proposals/simulations/`. A group contains four fresh attempts on one task; ten groups form a target batch. The sampler excludes all earlier task identities in the same optimizer step, including discarded refill candidates.

## Plan of Work

First implement a deterministic engine and checks for discovery accounting, task uniqueness, retained-group bounds, and separation of hidden task behavior from controller observations. Next build an HTML player with a compact four-class inventory, per-task state colors, selection reasons, attempt outcomes, model-update phases, and replay controls. Finally embed its assets into the standalone v2 reader and record the toy model's assumptions in the Markdown.

## Concrete Steps

From `/home/hammad/projects/rl`, run `node docs/research/proposals/simulations/task-discovery.test.cjs`. Render with `uv run --no-project --with pypandoc-binary --with beautifulsoup4 python docs/research/proposals/render_curriculum_v2.py`. Check the renderer with Ruff and run `git diff --check`. Use a temporary jsdom dependency outside the repository for DOM interaction checks if native browser preview is blocked.

## Validation and Acceptance

Every step must have distinct candidate task IDs across rounds. Reserved unseen selections must follow the cumulative 20% ledger until inventory exhaustion, which must be explicit. Model probabilities can change only at a model update, never during refills. All-success active sampling must stop at its bound without a fictitious update. Play, pause, next event, next step, reset, replay position, scenario changes, and task inspection must work. The generated reader must contain all assets locally and retain valid navigation and math.

## Idempotence and Recovery

Re-rendering overwrites generated v2 HTML and the standalone simulation only. Source files remain editable, seeded runs are repeatable, and the original proposal remains unchanged. Preserve unrelated dirty worktree files. No external service, credentials, migration, or deployment is involved.

## Artifacts and Notes

The expanded engine test passed 33,808 behavioral checks in addition to quota arithmetic checks. It verifies identical starting tasks and first outcomes for the ability and signal-response presets, then different model updates from the assumed learning law. Running the same test with --dom and NODE_PATH=/tmp/curriculum-ui-check/node_modules also verified play/pause, next model step, task inspection, truth toggle, ordinary/stress menu grouping, replay scrub, reset, and bounded refill failure in both generated pages. The temporary jsdom installation is an authoring dependency outside the repository. Ruff, JavaScript syntax checks, and whitespace checks passed.

The original v1 files remain unchanged. The player engine, view, styles, and fragment are maintained separately under docs/research/proposals/simulations/; the renderer embeds them into task-discovery-v2.html and the v2 proposal. On 2026-09-12 the plan was completed with this synthetic-model boundary to keep the demonstration distinct from real trainer qualification.

## Interfaces and Dependencies

Expose a pure JavaScript engine to Node for tests and to the browser player. The engine returns immutable event frames; the view replays them without advancing hidden state. The renderer embeds CSS, the engine, and the player into both the proposal and a standalone HTML page. Dependencies are authoring-only; the reader requires no package installation.
