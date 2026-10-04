# Original Gmail message return evidence

Local candidate checkpoint, 2026-10-04. This increment is in the external
environment candidate, not an adopted dependency or published release.

`contracts/gmail_observations.py` introduces `GmailObservationSource` with
adapter `gmail.message_reads@1` and kind `read_message`. It retains only fields
actually returned by audited find/list/get and Gmail API GET handlers. Each
message has its original native identity and execution recipient. Native and
local result bytes and invocation arguments must agree; the native state write
must be acknowledged; returned identity, sender and body fields must match the
unique before-state message. The Gmail scope must not change during the read.

Minimal, metadata and ID-only responses do not acquire bodies from backend
snapshots. Actual returned fields govern admission: the installed REST get
handler returns full display content for its metadata option, whereas REST list
returns IDs only. A manifest can compare returned body and sender against a
frozen initial message to verify a successful full return. This is not proof
of comprehension or precise model token conditioning.

The adapter's 26 tests pass in 2.04 seconds. Root independently verifies its
source SHA `4afb2783681bc0f4e39e0f1eeeb491ba2a3d48f74404fc61089b170fec5dec9e`
and test SHA `98e23655cd32997482c4f4a2a3ad8054c06f635fd94e6c0a07d01fba275d3e74`.
Root schema/native obligation integration passes 136 tests in 3.15 seconds,
with scoped Ruff and Pyright clean. Combined adapter/schema/occurrence/native
transport regression passes 225 cases in 29.48 seconds. Public native replay and
seventeen alternatives pass in 4.23 seconds; the critic independently repeats
26 adapter and 17 public cases. The installed revision's combined focused gate
passes 196 cases in 6.98 seconds.

Final broad manifest/core/retained/public regression passes 1,230 cases in
49.27 seconds. Final scoped Ruff/Pyright and both repository diff checks pass.

The existing once-only obligation ledger also rejected mixed-revision reuse
after this increment. Its stable key is episode, check, candidate and channel.
A retained valid contribution consumes the obligation even if its lifecycle
later failed or was interrupted. A changed contract or selector must not award
the same obligation again inside that ledger; intentionally new scoring versions
require a separate replay ledger. Genuine async yield-then-failure/cancellation
and reload tests pass. The critic independently reproduced both lifecycle paths
and the revision-only rejection.

Astra's public Contact assistant revision preserves the full prompt, initial email
and Contact inventory bindings. It adds the explicit original-message Find goal
without changing the requested assistant fields or inventing an action order,
extra notification or mandatory summary. Whole-task acceptance remains open:
the conditional rule limiting what summaries may mention needs observable output
evidence when applicable. Successful read and Contact state alone cannot prove
that rule was obeyed. The archive retains SDK output, but the prepared native
source does not currently expose a qualified authored-output inventory. This is
a projection/support gap, not proof that no summary existed. No catalog count
changes at this checkpoint: the existing bounded component is upgraded, not a
new fully qualified task.

Installed `public_contact_read_v2` bytes equal the tested draft SHA
`cbefe0c493db76949cff181eff023396feb1eed47ecf449654ae9ceaea4f2c9a`.
Its public native test SHA is
`5015a3071efea28cd236c38dda30be985335e6707b34bc625f17e91fe5bd60ad`.
The original Luna episode SHA remains
`05c81896501fa361b1d35d1388d1e7300eee468b1aacd8b2cced895dafeee037`.
Read invocation `eb45e8e2c5c648c1bce3fdb9994f3c29` and Contact mutation
`8baf7cd9ff6549b49fd1f565a59ad93a` each receive +1 once across current-version
score/rescore/native reload. Original bytes, scalar scores, public initial state,
prompt and execution receipts remain unchanged. Unrelated actions leave complete
read-inventory coverage false; they do not erase the known read witness.

Legacy Contact evidence hydrates omitted unrelated timestamp defaults from the
current clock. Current-wave findings preserve subject/signal/value/reason, but
this does not qualify stable whole-view hashes. Keep that reproducibility debt
visible. Mixed-version protection in this increment covers the required-effect
policy; the preserved record-transition and other legacy policies still need a
separate migration audit. Fresh-original replay is mandatory for this upgrade.

Run focused qualification from
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

    PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_manifest_gmail_observations.py tests/test_manifest_contracts.py tests/test_manifest_obligation_assessments.py -q --tb=short

Preserve original Luna episode files. Re-score a freshly loaded original under
the same candidate declaration for rescore/reload testing; do not mix prior
manifest revisions into its consumed credit ledger.
