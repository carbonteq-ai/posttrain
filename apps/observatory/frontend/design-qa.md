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
