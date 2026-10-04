# Report design QA

Final result: passed

Source visual truth: `design-evidence/concept.png`, grounded in the user-supplied report screenshot. The user delegated design selection and implementation. This is an adapted report, rather than a pixel-identical implementation of the generated concept. Generated numbers and prose are not research evidence.

Implementation evidence: `design-evidence/opening.png`, `trajectory.png`, `strategies.png`, and `mobile.png`. Combined comparison input: `design-evidence/comparison.png`.

Desktop implementation viewport: 1280 × 720 CSS pixels, device pixel ratio 1; the browser screenshot output is 1265 × 712 pixels. Concept: 1385 × 1136 pixels. The comparison preserves each aspect ratio, reduces each to fit a 900-pixel column, and labels the difference. It compares hierarchy and the same vendor-payment case; it does not pretend different viewport proportions establish pixel fidelity. Phone check: 390 × 844 CSS pixels; browser screenshot output is 375 × 812 pixels. These output size differences are recorded rather than interpreted as layout drift. Default viewport restored afterward.

## Findings and changes

- The earlier report buried conclusions under long methods text and duplicated tables. The new opening leads with the finding and a navigation path. Exact tables, source identities and measurement limits remain available through disclosures.
- The first implementation retained an overly long title and centered the prose inside a wider region. The title was shortened and prose aligned to the section edge. `opening.png` is the post-fix capture.
- Strategy headings initially concealed their mechanisms. Each closed strategy now includes its first explanatory sentence. The local-cost strategy opens by default; pilot steps use separate disclosures. `strategies.png` is the post-fix capture.
- Original narrative block identities are retained when content is placed in disclosures, avoiding collisions between strategy blocks and preserving existing text-edit addresses.

## Required surfaces

- Typography: existing report sans-serif runtime retained. Large title, numbered chapter headings, 16px prose with narrower lines, and smaller source notes provide distinct reading levels. Opening and trajectory captures were inspected at their actual size.
- Spacing: navigation and chapter dividers establish the main flow; explanations use a 720px maximum width. Evidence uses the wider report area. Six turns fit a three-column desktop grid and a single-column phone layout.
- Colors: the existing system theme is retained intentionally. Semantic green indicates positive replay credit; amber marks the focal harmful turn. Score decline remains labeled numerically, so color alone does not carry the distinction.
- Images: generated mockups remain design evidence only. The report contains source-backed data views, rather than rasterized mockup text or invented illustrations. Token blocks are a compressed visualization of the native policy mask, with the compression explained in the legend.
- Copy: all empirical values come from the report snapshot and retained trajectory projections. The concept's invented intermediate rewards were discarded. Historical actions, current-recipe credit replay, counterfactual costs and unmeasured learning effects are distinguished.

## Browser verification

Navigation to the actual trajectory and strategy section worked. Strategy disclosure closed and reopened. Selecting temperature 0.5 showed seven checkpoint rows, all at 0.50; the filter was restored to both temperatures. Seven native MathML equations remain mounted. No horizontal document overflow was observed on desktop or at 390px. The phone capture was inspected for readable wrapping and a single-column turn grid. No browser console errors were reported.

The combined concept/implementation comparison was inspected, along with full-size opening and trajectory captures. The trajectory capture itself provides the focused comparison: action labels, native token counts, credit signs and focal score drop are readable. Intentional deviations are the preserved system theme, compact turn grid, removal of decorative icons, and corrected source-backed values. No remaining actionable P0/P1/P2 issue was identified within these checked states.

## Limits

This pass verifies presentation and the checked interactions, not the causal research claims or an online training improvement. It does not exhaustively test every report edit or every browser. The standalone export passed the Data app export validation; its appearance shares the same authored modules as the checked local preview.

## Real-turn strategy explanations, October 2 update

Visual reference: [Trajectory Judge Bake-off](https://claude.ai/artifact/1ZRmDgY6CV9Lu7nfE69qFo), inspected in the in-app browser. Its short Thinking / Action / Result rows guide the presentation only. Its judge results are not imported as evidence for the SAMPO diagnosis. Reference capture: `design-evidence/turn-layout-reference.png`. Combined comparison: `design-evidence/turn-layout-comparison.png`; implementation capture: `real-turn-excerpt.png`.

The updated native projection includes shortened recorded reasoning, tool arguments and subsequent tool observations. Six vendor turns and ten advertising turns retain their exact native sampled-mask counts and source hashes. The turn selector defaults to the diagnostic turn and exposes the rest without printing the full transcript.

Mask evidence: `design-evidence/real-turn-mask.png`. The same real Turn 3 appears before and after the proposed intervention. Thinking and Action rows are outlined; Result is not. The policy-support strip is positive on the current-recipe side and negative under replacement. The strip compresses the full turn mask; it does not claim eight individual original tokens or precise call-token attribution.

The first two-column mask view was too tall to inspect the action and credit together. Excerpts were shortened further for this comparison; the full default turn view retains the longer abridged excerpt. The post-fix capture shows thinking, action, result and both credit signs together. The evidence preserves readable existing typography, full report width, theme tokens, an amber mask boundary, and explicit numeric credit signs. No decorative assets were added.

Interaction checks:

- Replacement gives −0.03333 and subtraction +0.10714, compared with +0.14048 current credit.
- Local turn weight 4 leaves singleton local credit at zero and combined credit positive.
- The six actual vendor siblings recompute episode credit after the declared 1.125 guard cost; both safe siblings and all four violating siblings are visible. Common rescaling is explicitly excluded from this table.
- Four allocator rows render from the report's recorded-bank table. Practice stages distinguish the recorded complete branch, unresolved admission, and proposed fresh rechecks.
- Linear length weighting makes the per-token factors equal while increasing the 13,011-token advertising episode's total influence by 5.99× relative to the 2,173-token vendor episode. Equal advantage is a controlled assumption, not a measurement of the traces' actual gradients.
- Switching the vendor view to Turn 4 shows the mistaken row mapping in reasoning and the actual Acme/CloudHost tool results; switching back restores Turn 3.
- Seven MathML equations remain mounted; browser console errors were empty.
- At 390 × 844 CSS pixels the comparison stacks into one column, has no document overflow, and keeps the actual call text readable. Capture: `design-evidence/real-mask-mobile.png`. Default viewport restored.

Final result for this update: passed. The reference uses a different task and typography; those are intentional differences. We adopt its evidence-row structure while retaining our own recordings, report theme and interactive strategy controls.
