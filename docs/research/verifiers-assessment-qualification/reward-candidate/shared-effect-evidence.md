# Shared deterministic effect evidence

The isolated environment candidate adds `effect_evidence.py`. It separates
captured world evidence from business success: a returned operation with valid
state acknowledgements can be a read, idempotent write, failed business command
or actual mutation. Task rules decide accomplishment and policy permission.

The adapter preserves distinct native invocation identities, rejects duplicate
terminal observations/acknowledgements and ambiguous captures, verifies snapshot
hashes, and requires matching explicit integer state revisions. Missing capture,
pending execution, write conflict and unsupported origins remain unavailable
with separate reasons. Existing-object transitions are visible, including draft
to sent; no object creation requirement is imposed by this shared layer. It does
not infer complete observation coverage from an empty set of transitions.

Twelve tests pass, including actual retained offboarding receipts, repeated equal
payloads with distinct invocation identities, failed local operations, existing
object transitions, prefix immutability, missing revisions and corrupt snapshots.
Together with the existing HR and revision identity suites, the earlier combined
run passed 42 tests before the final two prefix/duplicate-ACK tests were added.
The final focused effect suite passes all twelve; Ruff and focused Pyright pass.
The Mailchimp declared-set serializer fix passes two additional tests, including
four independent process hash seeds. It sorts only the declared `tags` set,
preserving ordered `notes`; old collection records are not rewritten.

These are environment-internal working values, not another persisted API. Native
assessment source identity remains mandatory at the call site. Tuple order does
not establish a total order of concurrent effects. Domain adapter integration
and all-task guard coverage remain open. The new Mailchimp serialization is for
a future source revision and does not override failed exact historical replay.

Run from `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

```sh
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src .venv/bin/python -m pytest tests/test_effect_evidence.py tests/test_mailchimp_serialization.py -q
.venv/bin/ruff check src/automationbench_v1/effect_evidence.py tests/test_effect_evidence.py tests/test_mailchimp_serialization.py
.venv/bin/pyright --pythonpath .venv/bin/python src/automationbench_v1/effect_evidence.py
```
