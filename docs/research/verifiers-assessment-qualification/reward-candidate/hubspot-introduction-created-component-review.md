# HubSpot introduction: created-contact component

Status: **uninstalled development candidate; bounded offline native proof passed**. This is not whole-task qualification, model semantic qualification, or catalog acceptance. No production, native, environment test, catalog, or original rollout files were changed.

## Public meaning and manifest

The frozen batch02 public input for `simple.email_hs_create_contact` asks to find Nathan Brooks's introduction and create a HubSpot contact with his information. Public input SHA256 is `e680e005042d6947911be37db14d7ec503861243e54c97ac785742cb2edd91fa`.

The adjacent `hubspot-introduction-created-component-draft.json` contains only the creation component. One `public.request@1` obligation declares these native contact fields:

| Field | Value | Public authority under `task_evidence.initial` |
|---|---|---|
| email | nathan.brooks@pinegrove.example.com | Copy `gmail.messages[0].from_` |
| firstname | Nathan | Reviewed literal from `gmail.messages[0].body_plain` |
| lastname | Brooks | Same introduction body |
| phone | +1-555-4242 | Same introduction body |
| company | PineGrove Analytics | Same introduction body |
| jobtitle | Head of IT | Same introduction body |

The original message identity is `msg_3042`, thread `thr_3042`. Its introduction explicitly supplies the name, title, company and phone. The manifest binds the full public prompt (`d8236c62e2676cdaf10d0fbacd4b6149fae7aed06200aa5f409e7fe0f25a0439`) and raw public initial inventory (`feb36f2cebb03577a624e7d27df38d77ff3690129782b7f90d5e823dc2308281`). It does not bind model-hydrated timestamps as policy or derive policy from the solver's arguments, hidden assertions, or final state.

The shared `hubspot.objects@1` adapter supplies native top-level object fields. The shared created-object evaluator privately projects those as `retained.fields.*`. Six equality checks require a fresh matching contact retained at completion. Credit is separately declared as `created_retained_completion_once@1`, earliest selection, with exactly those six goal fields. Generated IDs, timestamps, lifecycle stage, custom properties, company-object creation, outreach and subscription are not requirements.

This is a faithful-copy qualification slice: exact phone formatting and the reviewed first/last-name split are accepted. It does not prove general semantic equivalence for alternate phone formatting or paraphrased titles. Other initial-contact populations require a newly reviewed public binding; this declaration does not decide an unspecified duplicate/upsert policy.

Candidate file SHA256: `d2d9a32d1cee50368ee811c8a738327747b3e6b8ef95d79528ce0ea80ffe08ed`.
Canonical contract digest: `5e2d3a4037925639dfa814088ba32106a668e171b8909cc866b95589105b27a0`.

## Original native proof

Original episode: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/luna-reference-campaign-01/remaining700/untimed/f621d9c26a34cbe539178d8c72a90d9d469923166e1e5cda9c301dab3b1e1998/episode.json`.
SHA256: `642d1e4f0fb932bf9962fe52c8784315553882f5c3ccdf7432220cdae67745c9`.
Trace: `69a17cb7608c4692a8b8526da4759d10`.

The proof script loads this immutable episode and its retained artifacts, attaches the original public initial and retained final state using the existing replay pattern, and invokes **`ManifestAssessmentTask.score_assessments`**. This runs the real native assessment/credit lifecycle while bypassing the official legacy scalar scorer and all model/SDK backends. The candidate loader is injected only within this isolated proof process; the installed catalog is untouched.

First score, same-candidate rescore, supported WireEpisode JSON reload, and another rescore all retain:

- Goal `simple.introduced_contact_retained = 1`, reason `created_matching_fresh_object_retained`.
- Separate coverage diagnostic `1`, meaning only this created-object scope is closed.
- One completion contribution `+1`, to original physical invocation `8b410cc1e6944249ac2ab865ca3d453a`, revision `5→6`.
- Fresh retained contact `6669431507`. Adapter `receipt_id` is the invocation ID, not this contact ID.
- Exact same credit-assignment ledger after rescore/reload, unchanged original scalar reward, original execution events/state-write receipts and task source material.

The original ledger was empty; no retained contribution was erased. A conflicting manifest revision on the already-credited detached replay produces `credit_planning_failed:ValueError` and leaves the ledger/scalar unchanged. A separately loaded original replay with a different, typed public request produces abstained goal and coverage findings with `manifest_source_binding_mismatch`, and no credit.

Both original episode and artifact-envelope bytes remained unchanged. All **868** fingerprinted environment source/test and native-source Python files matched before/after. The proof contains their exact hashes and the full original recipient evidence.

### Reload compatibility limitation

Literal `WireEpisode.model_validate_json(archive, strict=True)` currently fails with 123 tuple-type errors after archive restoration converts JSON arrays into Python lists. Global strict Python re-admission of the loaded owner's dump also fails for 16 pooled `batch.views` lists. These failures are retained in `acceptance.json`; neither is reported as a success.

The supported `WireEpisode.model_validate_json(archive)` path succeeds. Every restored standalone execution receipt, state-write receipt, assessment batch and credit assignment additionally passes **strict Python re-admission**, and native rescore preserves the ledger. This qualifies the supported reload path, not global strict owner/archive compatibility. No native repair was attempted.

## Additional bounded fixtures

Five offline cases use genuine installed HubSpot handlers and existing development capture helpers, not actual host-issued transport:

| Case | Result |
|---|---|
| Canonical six-field create, a newly generated ID | Goal1 and qualified completion |
| `first_name` / `last_name` aliases | Goal1 and qualified completion |
| Two correct creates | One singleton completion selection, first qualifying invocation |
| Later real handler changes phone to `wrong` | Final goal0, no completion credit |
| Removed write ACK, retained correct final record | Goal1, no completion credit |

The damage fixture establishes terminal retention failure only: its manufactured update request omits the dynamically obtained ID, so that update transition is deliberately unqualified. It is not proof of update-action attribution. Fixture checks are pure shared-core evaluations; actual native once-credit qualification comes from the original episode above.

## Separate public obligations and open gates

The original `gmail_find_email` full return at invocation `1703ade1b55443d79b1c60e847974179` independently supplies `msg_3042`, original sender, and the complete introduction body. The existing read component can declare `new_occurrence` and `required_effect_once@1`; it is intentionally outside this creation-only draft and contribution count.

Whole-task acceptance still needs the original read obligation plus all applicable public policy guards. Current external-output capture leaves both `hubspot_find_contact` and `hubspot_create_contact` unsupported. A shared audited read/create-text classification must close that inventory before clean summary-exclusion or no-clarification compliance is claimed. All authored strings need preservation or explicit unavailability; a name/title/property is not automatically non-narrative. No mandatory final summary, prohibition on arbitrary extra outreach, exact turn-count gate, or SDK-assistant token recipient is invented here.

The summary and explicit no-clarification@2 policies require their own pinned current assessors and qualification. This component cannot certify their semantic behavior. Duplicate creation is not rewarded twice, but is not separately declared a public prohibition.

## Reproduction and retained artifacts

Proof directory: `/home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/hubspot-introduction-created-qualification-01/`.

From `/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`:

```bash
PYTHONPATH=/home/hammad/projects/verifiers-credit-candidate-20261003:src:tests uv run --no-sync python /home/hammad/projects/rl/.posttrain/state/verifiers-assessment-qualification/hubspot-introduction-created-qualification-01/qualify.py
```

The script starts from the immutable original each time and writes only this development proof directory. Re-running replaces derived files there; retain an independent copy first if comparing runs. No network or model calls are needed.

- `qualify.py`: SHA256 `95d9fa4a4d9989db50b9a9735128586f21e92a4c5c28ce222192ec26dc82e335`.
- `acceptance.json`: SHA256 `65727037860c53e1d6296938885b91a90d8a3932c3e460f20df3698fb033b5bc`.
- `episode-scored.json`, `episode-rescored-reloaded.json`, `episode-mixed-revision-rejected.json`, and `episode-public-retarget-abstained.json` retain native histories.
- `run.log` records the final successful execution. Earlier global-strict failures are recorded as structured compatibility findings, not silently discarded.

Native reload/credit and five handler/core cases are qualified only within the stated component scope. Independent parent/critic acceptance and catalog promotion remain separate.
