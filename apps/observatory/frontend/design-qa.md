# Observatory visual QA

Date: 2026-07-22

Status: passed

## Visual truth

The comparison sources are the accepted light-theme explorations in
`docs/design/observatory/moodboard/`:

- `observatory-focused-run-brief.png` for the run workspace;
- `observatory-traces-and-evaluation.png` for trace-derived evaluation;
- `hex-homepage-light-texture-reference.png` for the warm, restrained shell.

The sources are 1487 by 1058 pixels. The final desktop implementation captures
use a 1538 by 1226 browser viewport. Real schema-v4 Trackio data is the visual
truth for the SFT and system-metric workspaces. A deterministic evaluation
fixture remains the visual truth for the trace workspace because the two real
training runs do not emit Verifiers traces.

## Full-view comparison

The final evidence is retained under
`artifacts/observatory/validation/visual-audit/`:

- `13-real-v4-sft-overview.png` is the final SFT focused brief over a real QLoRA
  run;
- `14-real-v4-sft-system-metrics.png` is the normalized Trackio system view;
- `15-trace-filter-popover.png` is the final trace/evaluation workspace with
  the compact slice filter open;
- `16-real-dpo-artifact-lineage.png` is the real DPO consumed-artifact → run →
  produced-artifact lineage view;
- `17-compact-run-navigation.png` is the corrected evidence navigation and
  control typography after removing the global font-size override;
- `12-real-v4-dpo-overview.png` is the real DPO preference-learning view;
- `05-narrow-sft-overview-fixed.jpg` is the corrected narrow layout;
- `10-typography-theme-final-sft.jpg` and
  `11-typography-theme-narrow.jpg` are the final hierarchy/theme pass;
- `01-sft-overview.jpg` and `04-narrow-sft-overview.jpg` preserve the before
  states for the fixes below.

The implementation keeps the visual hierarchy of the sources: a stable evidence
shell, warm off-white surfaces, editorial headings, thin separators, one
dominant scientific figure, compact supporting values, and a contextual
inspector. It intentionally does not reproduce the source's dense secondary
navigation, comparison controls, or decorative line work where those elements
do not yet correspond to working product behavior.

## Findings and corrections

1. P1 - The first SFT pass used the right rail for selected-step and run-detail
   metadata, leaving no continuous input-to-output story. The final view keeps
   the exact selected step and values in a chart-local strip and dedicates the
   right rail to resolved model, dataset, training binding, and produced
   artifacts. Evidence: `01-sft-overview.jpg` compared with
   `13-real-v4-sft-overview.png`.
2. P2 - The initial narrow layout allowed the desktop canvas to produce global
   horizontal overflow. Fixed by making the shell and evidence grid shrink,
   retaining project/run/status context, and containing section navigation in
   its own horizontal scroller. Evidence: `04-narrow-sft-overview.jpg` compared
   with `05-narrow-sft-overview-fixed.jpg`.
3. P2 - The trace summary label `Failed` visually conflated ingestion failures
   with verifier review outcomes. Fixed to `Errors`; verifier outcomes remain in
   the charts and trace table.
4. P3 - The implementation is slightly less decorative than the mood-board
   source. This is accepted for the first product slice: texture stays in quiet
   shell regions and never reduces chart, table, or transcript legibility.
5. P1 - Supporting text was initially compressed into 8-11 pixel styles while
   display values jumped directly to very large serif sizes. Fixed with shared
   eyebrow, page-title, page-subtitle, label, body, and dense-table tiers. Run
   names and breadcrumbs now sit between navigation and metadata instead of at
   metadata size. The 375 by 812 check retained zero page-level overflow.
6. P2 - Neutral colors were embedded throughout individual components. Fixed by
   introducing semantic light-theme tokens for canvas, panel, surface, subtle
   surface, divider, ink, secondary text, muted text, and accent. Components
   now consume those roles while chart-series colors remain a stable evidence
   palette.
7. P2 - Native select controls made trace filters look unrelated to the rest of
   the evidence system. Replaced with compact Radix popovers using the same
   control height, typography, focus treatment, and semantic surfaces as the
   chart controls. Evidence: `15-trace-filter-popover.png`.
8. P1 - Fixture-only review concealed provider-specific metric names and empty
   lineage fields. Real schema-v4 SFT and DPO captures now prove the hierarchy
   with resolved QLoRA bindings, dataset revisions, output artifacts, TRL
   metrics, and normalized Trackio hardware telemetry.
9. P1 - The first Artifacts & lineage page was only an artifact list, so the
   run-centered consumed/produced relationship was not visible. Fixed with an
   explicit three-stage lineage view backed only by recorded artifact edges,
   including producer metadata for consumed artifacts and truthful empty output
   state for failed runs. The provider/version/digest ledger remains below the
   graph. Evidence: `16-real-dpo-artifact-lineage.png`.
10. P2 - The evidence-section navigation declared 12-pixel type but rendered at
    16 pixels because a late global `font: inherit` shorthand overrode Tailwind
    font-size utilities on buttons and inputs. Replaced the shorthand with a
    family-only reset, then reduced the section bar to 36 pixels with 11-pixel
    labels and 16-pixel gaps. Evidence: `17-compact-run-navigation.png`; browser
    computed styles confirm 11-pixel labels and a 36-pixel bar.

## Detail checks

- Typography: serif display headings and tabular evidence values reproduce the
  editorial/analytical contrast without sacrificing dense UI readability.
- Layout: the run brief preserves one dominant scientific figure, a compact
  selected-step strip, and a lineage rail; the trace page preserves aggregate
  context, a dense evidence table, and a selected-trace inspector.
- Color: the warm neutral canvas and restrained violet, teal, coral, and green
  accents remain consistent. Status is not communicated by color alone.
- Charts: plot canvases are untextured, axes and grids are thin, tooltips and
  zoom are available, and chart content has a textual screen-reader summary.
- Tables: trace rows are real semantic table rows and remain sortable while the
  body is virtualized.
- Copy: headings describe the expert question or evidence relationship rather
  than generic dashboard categories.
- Assets: interface symbols come from Phosphor Icons; no placeholder or
  handcrafted iconography remains.

## Interaction and runtime checks

The Codex in-app Browser verified the real Trackio SFT and DPO run selection,
job-specific chart selection, the real system-metric projection, and the
fixture trace workspace. The slice popover was opened, `calendar` was selected,
and the table correctly changed from 12 of 12 to 8 of 12 traces. No browser
developer-tools integration was used for the final validation.

Final result: passed.

## 2026-09-23 controller evidence chart revision

Source visual truth: `/home/hammad/.codex/generated_images/01a0ce24-5d5b-73d0-bcfd-1c85fec9e39d/exec-99c3b924-b638-4eb5-be4a-7e43a077d78c.png` (1586 × 992 px, selected concept 3 revised with a step-range slider and near-contiguous bars). The live implementation was captured inline through the Codex in-app Browser on `http://127.0.0.1:7871/` at 2550 × 1226 px; that browser capture was not exported to a filesystem path. Comparison focused on the controller-evidence panel because the source mock shows 20 illustrative steps while the real run currently has 13 steps. This content difference is expected; both use the same desktop layout, light theme, and selected run route.

### Findings and comparison history

1. P2 - The first rendered pass lacked candidate totals above the stacked bars. The generated source places totals over each step. Fixed by aligning a numeric label row to the chart's category columns. The second live browser capture shows totals above all 13 bars.
2. No remaining P0/P1/P2 findings in the target panel. The final capture shows two aligned data views, class colors retained from the prior UI, bars separated by approximately 2% of one category width, and a compact outcome matrix. The panel is shorter than the old full-width row list at this data size. The first and last range handles are visible and do not obscure the chart.

### Fidelity surfaces

- Typography: existing Instrument Serif heading and Inter evidence labels are preserved. Candidate totals and matrix values use tabular numerals; the mock's small-label hierarchy is retained.
- Spacing and layout: one slider sits above the legend and chart; the outcome matrix follows immediately below. The chart and matrix share a 96 px left label/axis inset. The browser capture shows no clipped content in the target panel at the checked desktop viewport.
- Color: the seven recorded class colors remain stable; the slider uses the existing violet accent. The matrix uses distinct muted markers alongside numeric values, not color alone.
- Image and assets: the target contains data charts and existing UI icons, with no new image asset to reproduce. The distribution is rendered from live values rather than a raster mock.
- Copy and content: the panel title is retained, and the visible range reads `Steps 1–13 of 13` for the current run. The default follows the latest 20 when at least 20 steps exist; with fewer steps, it shows all available steps.

### Interaction and validation

The browser showed the real `lfm26-vortex-v3-lr2e4-20260923-r1` evidence. Dragging the first handle changed the view from steps 1–13 to 2–13; keyboard Right made the same change, and `Latest 20` restored steps 1–13. A component test verifies the initial 6–25 window for 25 steps and moving both bounds. TypeScript check, 82 frontend tests, production Vite build, and `git diff --check` passed. The browser console had one older dynamic-import error while assets were being rebuilt; after the final reload the page and chart rendered and no new error appeared. Narrow-screen visual QA remains a follow-up check.

Final result: passed.

### Class-order follow-up

The class stack now uses descending candidate count across the full run, with alphabetical ties. This same order is applied to every step and does not change with the range slider. Colors remain assigned by class identity, so changing the range cannot recolor a class. A component test covers a class that dominates the full run but is absent from the default latest-20 window. The live browser check on the 13-step run showed one consistent legend and band sequence across all columns; segment heights and boundaries still vary with the observed class shares. TypeScript check, all 83 frontend tests, production build, and `git diff --check` passed.

### Prompt-group investigation follow-up

Latest-first and aligned-expansion update (2026-09-23): Training trace pages now request newest recorded summaries first; other trace views retain their previous default order. The local Trackio-backed preview returned optimizer step 13 in its first five records instead of step 1, and a later live refresh displayed step 14 groups at the top of the first 100. Group rows are shown newest first while previous reward/variance is calculated chronologically from complete loaded groups. Page offsets on a running source are not a frozen snapshot; repeated trace IDs across pages are removed client-side, and group statistics remain limited to loaded records. Expanded rollouts now occupy child rows in the same HTML table and share its seven-column grid. The live browser accessibility tree showed a group row followed by four rollout rows, each with an aligned reward cell and trace-selection button. Frontend tests (86), TypeScript check, production build, and targeted backend tests (91) passed. Repository-wide pyright remains blocked by a type error in `packages/train/tests/test_adaptive_curriculum.py:773` outside this change.

The annotated trace table lacked four pieces of context: optimizer-step filtering, exact within-step prompt grouping, update-selection state, and prior reward spread. The revised preview on port 7872 groups only by the recorded `posttrain_prompt_group_id`, projects the recorded `optimizer_step`, offers a step filter over loaded summaries, and shows reward mean and population variance only for complete scored groups. Previous mean and variance refer to the most recent complete group for the same task among loaded summaries. In the browser, choosing step 2 reduced the visible groups from 28 to 9; a partial group displayed `2 / 4` and suppressed its current statistics. The table explicitly says `Not recorded` for update selection because this run has no per-group inclusion decision. Trackio's physical `step` was checked and differs from `optimizer_step`, so filtering physical steps would be wrong. The page stays provider-bounded rather than scanning the whole run to make a filter appear complete. Full-population step filtering and definitive retained/rejected status need additional indexed producer/provider evidence. The screenshot and interaction were inspected in the local browser; the browser capture was not exported to a file.
