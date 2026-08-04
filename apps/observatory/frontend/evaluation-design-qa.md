# Observatory trace investigation design QA

## Evidence

- Source visual truth:
  `docs/design/observatory/moodboard/observatory-traces-and-evaluation.png`
- Browser-rendered implementation:
  `docs/design/observatory/audit/evaluation/observatory-reference-aligned.jpg`
- Full-view comparison:
  `docs/design/observatory/audit/evaluation/observatory-design-comparison.png`
- Focused table and inspector comparison:
  `docs/design/observatory/audit/evaluation/observatory-table-comparison.png`
- Source pixels: 1487 x 1058.
- Implementation pixels and CSS viewport: 1839 x 1219 at browser density 1.
- Normalization: both full views were proportionally reduced to a maximum width of 1400 pixels before being placed side by side. The focused comparison crops the trace table and inspector from each source, then scales each proportionally to a maximum width of 1300 pixels.
- State: the source uses a math evaluation and the implementation uses the real 200-trace AutomationBench run. Content values differ intentionally; evaluation hierarchy, control placement, table grouping, density, and inspector structure are the comparison targets.

## Findings

No actionable P0, P1, or P2 differences remain for the requested trace-investigation hierarchy.

- Fonts and typography: the implementation retains Observatory's existing Instrument Serif and Inter system. Display hierarchy, compact table labels, numeric weight, truncation, and small-label tracking are consistent with the reference while preserving the product's established type tokens.
- Spacing and layout rhythm: summary, filters, three-chart row, and table-plus-inspector now follow the reference order. The table count is integrated into the card, declared metrics share a grouped header, and the table fits its primary evidence columns without a default horizontal scroll at the verified viewport.
- Colors and visual tokens: existing divider, violet-selection, teal-success, rose-failure, and amber-truncation tokens remain consistent. The table and its virtualized rows now use an explicit white surface, matching the reference instead of inheriting the warmer application canvas.
- Image and asset fidelity: the screen contains product UI and icon-library assets only; there are no missing raster assets or substituted illustrations.
- Copy and content: prompt preview now comes from the trace contract, trace IDs are visually secondary, metric labels remain schema-derived, and the inspector exposes both reward components and verifier signals.

## Comparison history

1. Initial P2: trace rows led with task identity and collapsed declared signals into a single text-heavy cell. Fixed by adding bounded prompt previews, a prompt-first column order, and individual schema-driven signal columns.
2. Initial P2: declared signals had no shared visual hierarchy and long AutomationBench labels collided. Fixed with a two-level `Reward components` header, compact leaf labels, bounded widths, and overflow containment.
3. Initial P2: filters sat below the charts and the trace heading floated outside the table. Fixed by moving filters above all derived views and integrating the trace count into the table card.
4. Initial P2: the inspector exposed only the primary reward component. Fixed by adding a declared verifier-signal section with the configured success signal's pass/fail mark.
5. Follow-up P2: the first grouped-header pass stacked two complete header rows, leaving empty cells above ordinary columns. Fixed with an explicit grid where stable columns use `rowSpan=2`, `Reward components` uses `colSpan=4`, and only its declared signal children occupy row two.
6. Follow-up P2: the table inherited the warm canvas surface. Fixed by applying an explicit white background to the table card, scroll viewport, table, virtualized body, and unselected rows.
7. Post-fix evidence: browser inspection confirms row spans of 2 for ordinary headers, a 4-column reward group, and computed table/body backgrounds of `rgb(255, 255, 255)`. The refreshed comparison images show the revised hierarchy and density. No P0/P1/P2 issue remains.

## Interaction and runtime checks

- Opened the real AutomationBench run and switched from Overview to Traces & evaluation.
- Selected a prompt-preview row and confirmed the inspector changed to that trace.
- Confirmed pass, fail, and truncated outcomes remain accessible by name while rendering as compact icons.
- Browser console errors checked: none.
- Frontend tests: 28 passed.
- Production TypeScript check and Vite build: passed.
- Observatory projection test: 22 passed.

## Follow-up polish

- P3: a future column chooser could expose response and thinking lengths without crowding the default evidence set.
- P3: pagination and next/previous inspector navigation can be added when server-side trace paging replaces the current bounded population.

final result: passed
