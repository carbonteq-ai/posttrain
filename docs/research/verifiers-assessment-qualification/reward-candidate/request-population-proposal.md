# Source-bound request population and the smallest complete Asana slice

Status: proposal only, 2026-10-04. No code, catalog, native contract or earlier review artifact changed.

**Recommendation:** add one source-bound request member, a bounded relation between two qualified effects, and the Asana evidence repairs needed to assess the complete simulator workflow. A request member by itself would only unlock partial creation checks.

Scope is AutomationBench's existing manifest engine. No task-specific Python, runtime prompt parser, judge, fake spreadsheet, ontology or workflow framework is required.

## 1. One authored request is a real population member

Add `RequestSource` in a small environment-owned `contracts/requests.py`, with adapter `public.request@1`. V1 contains exactly one named member, not arbitrary computed rows:

```json
{
  "adapter": "public.request@1",
  "member_key": "requested-task",
  "fields": {
    "name": {
      "value": "Refactor payment module",
      "authority_paths": [["task_evidence", "prompt", 1, "content"]]
    },
    "workspace": {
      "value": "ws_prod",
      "authority_paths": [["task_evidence", "prompt", 1, "content"]]
    },
    "project": {
      "value": "proj_eng",
      "authority_paths": [["task_evidence", "prompt", 1, "content"]]
    },
    "due_date": {
      "value": "2026-03-14",
      "authority_paths": [["task_evidence", "prompt", 1, "content"]]
    },
    "section_name": {
      "value": "Sprint 8",
      "authority_paths": [["task_evidence", "prompt", 1, "content"]]
    },
    "section_id": {
      "copy_from": [
        "task_evidence", "initial", "asana", "actions",
        "find_section", 0, "params", "section"
      ],
      "authority_paths": [
        ["task_evidence", "initial", "asana", "actions", "find_section", 0]
      ]
    }
  }
}
```

This is proposed syntax, **not a valid current ContractSpec**. Each authority path must refer to a matching existing `ContractSpec.bindings` entry. At least one binds a public prompt. Each field has exactly one strict scalar value or public-source projection; projections cannot reach final state, tool outputs, scores or hidden answers. The section source record must independently agree with the request's workspace, project and section name. Do not treat a hash match alone as that relational check.

The author translates the public request into typed data. A reviewed ISO date is an authored interpretation, not a date string extracted by a hidden task parser. Runtime admission checks source identity, types, projection and declared relations; it cannot prove that an arbitrary literal faithfully paraphrases prose. Semantic authoring review plus changed-parameter tests remains necessary. A `reviewed: true` flag would prove nothing.

The initial Asana record has `params.section=sec_sprint8`; its outer `id=as_rec_1` identifies the simulator lookup record, not the section. Copying the wrong one must fail qualification.

V1's indexed projection is conservative admission scaffolding: the full authority record is bound, so reorder or replacement rejects until rebound. It is not a generated-output ID or a hidden expected answer. An exact source-record selector can replace positional binding when variants require it; do not build a general query language first.

## 2. Reuse the structural population interface, preserve provenance

Return immutable `RequestPopulationEvidence` implementing the existing `Population` protocol: one member with projected fields, stable authored member key, canonical source/selector digests and its public authority references. Do not masquerade as `TableEvidence` or `InitialCollectionSource`.

A qualified request declares a closed population of **one authored obligation**, not a complete world population or proof that every sentence of the task is covered. Missing authority yields unavailable evidence, never a qualified empty population. The identity contains adapter/source identity and `member_key`; neither generated task IDs nor execution ordinals determine the obligation key. Retain the existing check/selector/member identity scheme across prefixes and rescoring.

Add a narrow admitted-source branch where `obligations._bind` currently requires every source path to start at `task_evidence.initial`. Do not weaken that rule to allow arbitrary contextual data. Extend the source union and native input capture/validation in `contracts/models.py` and `manifest_obligation_assessments.py`; rederive the request evidence from the bound raw source at the trust boundary.

**Authored request parameters are not observed baseline state.** V1 request goals use explicit `new_occurrence` semantics and cannot supply an `initially_satisfied_when` predicate over those parameters. Otherwise a manifest could turn an authored false value into manufactured initial non-satisfaction. A later mode may discharge a request from independently qualified initial-state evidence, with provenance-aware validation.

The simple Asana request explicitly asks to create a task. A similarly named initial object does not satisfy this new-occurrence requirement.

## 3. Join two effects under one obligation

Current `effects.required_when@1` consumes one effect inventory. Checking creation and placement independently permits “create task A; add unrelated task B” to pass. Joining generated identity is therefore necessary for this public workflow.

Add one bounded operation, conceptually `effects.required_pair@1`, with:

- the request population;
- first and second named effect sources;
- existing typed predicates for each effect;
- typed equality joins and an explicit native causal-order relation;
- a cap on candidate effect pairs.

For this task the first source is Asana `create_task`; match requested workspace, project, task name and semantic calendar due date. The second is `add_task_to_section`; match workspace, project and the resolved section. Require its `params.task_id` to equal the first effect's **generated task identity**.

Expose validated effect metadata separately from its parameter map, such as `first.witness.effect_id` and `second.params.task_id`. Preserve invocation, revision, source and selector binding for each witness. The section operation's own effect ID is a different action-record ID and is not the join target. Response template `gid` values are also not authority.

Require the first effect to causally precede the second using the authenticated revision/transition relation, not array position, string-sorted invocation IDs or guessed wall-clock order. Incomparable histories stay unavailable.

One request produces one assessment target and one result. Enumerating 100 possible pairs must not produce 100 obligations or rewards. An existential qualified pair can establish success; duplicate qualifying chains do not multiply reward. Unknown evidence prevents an absence claim but need not erase an independently complete positive proof. Multiple request members and cross-request ambiguity are deferred until a real task needs them; v1 has one member.

Do not choose the first creation before checking the join: a later valid creation/placement pair may satisfy the request even if an earlier unrelated creation does not.

## 4. Keep completion evidence separate from action credit

A pair result retains both supporting witnesses and identifies the placement as the completion witness. Add an explicit once-per-obligation credit selector for this result, bound to the same episode and contract. It may assign completion credit to the placement execution after the full pair is qualified. That is a declared domain credit policy, not proof that placement alone caused all value.

Do not pass both witnesses to today's `select_obligation_credit` unchanged: it selects the earliest witness, which would credit creation while treating joint completion as if it were a single effect. Preserve support witnesses independently from recipient selection.

Choose the earliest qualified completed pair deterministically, persist the obligation's consumption once, and reuse it on reload/rescoring. A new assessment snapshot, extra duplicate task or repaired presentation must not reset consumption. Changed contract semantics require explicit replay/version handling, not automatic fresh credit within the same training episode.

No native assessment API or archive store changes are needed. Existing source snapshots, views, result receipts, exact execution recipients and native archives remain authoritative.

## 5. Three Asana adapter gaps must be closed before “complete”

Source inspection found concrete gaps beyond manifest syntax:

1. **Missing buckets and the legitimate read.** The actual public initial world has only a `find_section` bucket. Current `contracts/effects.py` treats absent creation/placement buckets as incomplete scope and treats `asana_find_section` as unsupported. Consequently missing steps can remain unknown even with otherwise complete capture. Qualify the installed schema's `actions.setdefault(kind, []).append(...)` semantics for an absent bucket only when Asana and its action map are explicit and complete execution accounting is available. Missing service/actions remains unknown. Qualify the installed find operation as a scoped read, including result identity, without rewarding a read merely for occurring.
2. **Equivalent due-date arguments.** The installed create tool accepts `due_on` and stores it as `dueDate`. Current `asana_evidence.py` compares raw invocation keys to stored params and may reject this legitimate alias. Normalize only aliases declared by the installed service capability, then verify invocation/result/persisted agreement. Also admit equivalent supported date representations by declared calendar-date semantics, not a single teacher spelling.
3. **Occurrence versus retained task outcome.** The simulator stores action records; it does not expose a real Asana task/membership model. A complete simulator contract can require the matching creation and placement records to remain supported at the final capture, with their requested attributes intact and no unresolved relevant mutations. It must not claim external Asana membership. Do not turn two historical occurrences into retained success after known deletion or contradictory later mutation. Unsupported relevant mutation makes retained completion unavailable.

Resolve Sprint 8 through the scoped public section relation or a qualified lookup result; do not enforce an exact search transcript or extra lookup solely for reward. If literal execution of a lookup is adopted as a process requirement, record and qualify that requirement separately rather than smuggling it into the generated-ID join.

A proven wrong/missing step is failure only with adequate scope. Capture gaps are unavailability. These remain distinct even when the full happy path passes.

## 6. Implementation order and acceptance

1. Implement request source/evidence and integrate capture, revalidation and stable instance planning. No task installation yet.
2. Implement the bounded pair evaluator and separate completion credit selector. Register its source requirements as the union of request authority, both effect inventories and any retained-state evidence.
3. Repair/qualify the Asana adapter boundaries above, then install the data-only manifest and run actual native Task.score and reload tests.

Required tests:

| Boundary | Acceptance |
| --- | --- |
| Request authority | Wrong literal caught by authoring test; changed/missing prompt hash, hidden/final-state projection and duplicate/empty member key rejected. Malformed copied evidence cannot bypass recapture. |
| Parameter variants | Changed public task name, workspace, project, due date and section ID work after reviewed rebinding without Python edits. Omitted or conflicting scope stays unavailable. |
| Complete workflow | Actual simulator create + correct section placement produces one complete request outcome and one declared completion assignment. |
| Generated identity | Create A/add B, outer lookup-record ID, response-template gid, wrong project or same-name section in another project cannot pass. |
| Valid alternatives | Supported dueDate/due_on spellings and date representations; irrelevant extra calls; first unsuitable creation followed by a complete correct pair. No imposed exact prose or tool count. |
| Missing steps | Complete capture with no create/no placement yields failure; absent service, missing ACK, incomplete relevant history and partial-order ambiguity yield unavailable. |
| Retention | Correct historical pair followed by deletion, altered requested attributes or contradictory relevant mutation cannot remain retained success without a separately qualified repair outcome. |
| Replay and credit | Repeated matching pairs, scoring twice and serialized native reload do not multiply assignments. Support witnesses and recipient execution remain distinct. |
| Bounds | Pair budget overflow abstains explicitly; planning cannot silently truncate the candidate set or hide unexamined alternatives. |

## Alternatives and source anchors

Putting constants directly into the existing check is smaller syntactically but leaves no request member, public provenance or stable obligation identity. Reusing a fake sheet is semantically wrong. Making every effect a population member makes IDs/retries determine reward inventory. A general executable workflow graph is unnecessary for this two-effect task. The bounded request plus pair operation is the smallest coherent increment.

Inspected source in the environment candidate:

- `src/automationbench_v1/contracts/populations.py`: structural Population protocol; current initial-record registry is Gmail messages and Salesforce opportunities.
- `contracts/obligations.py`: initial-only binding, one-source matching, stable instance planning and earliest-witness selector.
- `contracts/models.py`, `manifest_obligation_assessments.py`: source admission and native lifecycle integration.
- `contracts/effects.py`, `asana_evidence.py`: separate qualified Asana occurrences and conservative completeness.
- Installed `src/automationbench/tools/zapier/asana/actions.py` and `src/automationbench/schema/asana.py`: argument aliases, returned action records and bucket creation semantics.
- `tests/test_manifest_effects.py`, `tests/test_asana_evidence.py`: distinct creation/placement IDs and capture qualification.
- Frozen batch04 public input SHA is retained in `manifest-authoring-batch-04-review.json`; its pack SHA is `08fa811e527fb7cf01213ae2f3424e9ec2dcdcb6f33618e74f4f3499aee963c8`.

Repository root for the implementation sources above:
`/home/hammad/projects/verifiers-environments-reward-candidate-20261003/environments/automationbench_v1`.

This is a design recommendation supported by current source inspection, not executable qualification or a whole-task completion claim.
