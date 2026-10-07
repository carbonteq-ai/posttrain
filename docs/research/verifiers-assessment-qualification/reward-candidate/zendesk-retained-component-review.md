# Zendesk retained-status component

The public request directly requires the existing ticket **ZD-501** to have status **solved**, and separately requires a password-reset resolution email to **elena.voss@retail.example.com**. The proposed manifest checks only the retained ticket status. No mutation attribution, action credit, email completion, or whole-task qualification is claimed.

The initial ticket has native ID ZD-501, subject `Cannot reset password`, status `open`, and priority `normal`. Subject, priority, a public closing comment, and an exact email wording are not additional required mutations. The request specifies no mandatory call order. There is no material ambiguity in the status component; notification purpose remains an independent unimplemented obligation.

The complete public prompt supplies both the selected native ID and desired status. Bind that prompt and the complete relevant initial ticket inventory for this reviewed declaration. This makes disappearance of the publicly identified target an explicit source-binding mismatch, rather than a vacuously satisfied empty population. Initial-inventory variants require renewed binding review; terminal records remain mutable observations. The shared operator itself remains population-based and must distinguish legitimate inapplicability from success. The source adapter must preserve canonical native identities without filling missing IDs or status from model defaults.

## Evidence and boundaries

- Frozen batch 10 exact SHA: `a99c6aa6fca047d97221b28679aec5811c74e9c7e651d352c8b5f4161d53e42c`; matched the authoring index.
- Public input canonical SHA: `3899a605929cffcb4a298bbc1b9ad546453b42dbc6eace69cb4cacf66241f0a3`.
- Complete prompt canonical SHA: `403ff6ea73310752e3f404979663e0120f7bc17fc18cbe5839c484d4705c9d02`.
- Retained episode exact SHA: `e04a068ad087b52ff28c8294f99cb539cf7f282c0914a795f654a115bd4439c9`.
- Trace: `53ebbeba533d417183fda46331d2c5d2`, completed, okay, no trace errors.
- Actual final collection: `traces[0].info.automationbench.end_state.zendesk.tickets`. Its uniquely identified ZD-501 has `status: solved`. These recorded facts support replay qualification, not policy authoring.

The installed simulator genuinely mutates `WorldState.zendesk.tickets`: `automationbench/tools/zapier/zendesk/tickets.py::zendesk_update_ticket` resolves the native ID, updates status, and returns the ticket ID and display record. The native schema is `automationbench/schema/zendesk.py::ZendeskTicket`. Its generated-ID and default-status behavior must never be used to manufacture omitted evidence during assessment.

## Qualification gates

The data-only [draft](zendesk-retained-component-draft.json) uses `records.retained_when@1` and has passed schema admission, compilation and serialization roundtrip. No draft is installed during this review. Require one outcome instance for the selected original native ticket, with explicit unavailable scope if its initial identity cannot be established; an absent target must not vacuously pass through an empty population.

Real-handler fixtures must cover solved, open, solved then reopened, reopened then repaired, no-op, and another solved ticket. Evidence counterexamples cover missing/duplicate/boolean IDs, missing status, missing or unfinished final capture, original deletion and replacement by a different ID, reordered collections, source-binding drift, and forged retained projections. Missing ACK must not by itself erase a valid terminal outcome, and no fixture may acquire action credit.

The actual episode gate must verify its exact bytes, preserved scalar rewards, outcome semantics in each new assessment wave, empty credit assignments, native archive reload, and unchanged source file bytes. A returned tool result or action log without retained status cannot satisfy this component.

Current status: **34 tests pass in 2.98 seconds**, including the historical replay, rescore, and native reload. The actual retained component outcome is 1, with zero action-credit assignments. Original episode bytes and scalar rewards remain unchanged; each new scoring wave preserves the same outcome/view/output semantics. Ruff, scoped Pyright and diff checks pass. Security checks reject a forged terminal projection, a self-consistent alternate raw source, and a direct assessor call without the native executor's sealed source. Both declaration bindings were independently recomputed from public inputs.

The first full run had 33 passes and one pre-scoring test failure: native prompt message models were compared directly with public dictionaries. Comparing the native data JSON projection fixes that fixture assertion while preserving exact prompt equality. The successful full rerun used the same frozen shared source; no source repair was needed.

Exact qualified hashes are retained in the JSON companion. The draft is uninstalled and awaits independent review/root acceptance. This proves the bounded status component only; email purpose and action attribution remain open.
