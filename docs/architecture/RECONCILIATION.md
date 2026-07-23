# Architecture reconciliation gap list

**Date:** 2026-07-21
**Authority:** [docs/post-training/](../post-training/README.md) (frozen baseline)
**Scope:** docs only — map the new model onto packages without rewriting product intent.

Existing `docs/architecture/*` and lifecycle ADRs remain useful engineering notes but
are **stale** until rewritten against this list. Prototype code is evidence only.

---

## Vocabulary map (old → new)

| Architecture / ADR language | Baseline |
| --- | --- |
| job + action + invocation | work package → job kind/definition → run |
| profiles (model / engine) | model variant + inference binding (`engine` field) + catalog |
| code-first only | config-first: YAML+schema **and** typed Python |
| Trackio group ≈ `job_id` | group ≈ `work_package_id` (+ `project_id`) |
| stages / lifecycle as platform spine | `screen` \| `train` \| `qualify` (selection is not a stage) |
| organization-shared assets | framework-shared catalog |
| EvaluationProgram as primary product noun | evaluation plan (+ environment binding) |

---

## File dispositions

### Platform overview

| File | Disposition | Notes |
| --- | --- | --- |
| [architecture.md](../architecture.md) | **rewrite** | Keep MVP boundary and package list; replace job/profile spine with Project → Work package → Run; point APIs at [05](../post-training/05-apis.md); observation at [06](../post-training/06-observation-and-lineage.md). |
| [architecture/README.md](./README.md) | **rewrite** (light) | Already links baseline; update table questions to new nouns; keep as index after concern docs land. |

### Layers and ownership ← 04 + 05

| File | Disposition | Notes |
| --- | --- | --- |
| [layers-and-ownership.md](./layers-and-ownership.md) | **rewrite** | **Keep:** `train`/`eval`/`serve` do not import each other; `data`/`common` roles; env packages independent; host injects observer. **Change:** “profiles” → catalog selections; lab “jobs” → work packages + job defs; public surface = primitive seats + job APIs from 05, not bag-of-kwargs operations. **Drop:** implying Job/Trackio are required for package reuse (already stated — reinforce). |

### Trackio / observability / lineage ← 06

| File | Disposition | Notes |
| --- | --- | --- |
| [trackio.md](./trackio.md) | **rewrite** | **Keep:** Trackio observational only; traces (Inference + Verifiers); artifacts; no control plane. **Change:** grouping key to `work_package_id`; envelope fields from 06 (`project_id`, stage, resolved seats, `source_layer`); metric namespaces `serve/*` `train/*` `eval/*` `system/*` `data/*` `tracking/*`. **Drop:** Git-as-sole-authoring story that excludes YAML catalog/overlays. |
| [observability.md](./observability.md) | **rewrite** or **fold into trackio** | Largely duplicates trackio + 06. Prefer one architecture doc after rewrite; otherwise thin pointer. Context today uses `job_id`/`action_id`/`invocation_id` — remap to work package / job / run. |
| [lineage-and-metadata.md](./lineage-and-metadata.md) | **rewrite** | **Keep:** artifact types, run↔artifact edges, decisions outside Trackio. **Change:** lineage = artifact edges (not package order); catalog refs + resolved snapshots; accept/revise/reject as project decisions, not job APIs. |

### Training / inference / evaluation ← primitives + job APIs

| File | Disposition | Notes |
| --- | --- | --- |
| [training-and-inference.md](./training-and-inference.md) | **rewrite** | **Keep:** package public ops list shape (`sft`, `grpo`, `launch`, …); optional host context; adapter internals. **Change:** ops take **resolved primitive seats** (02/05), not “profile + engine profile”; inference binding vs weight quant as new model variant; env owns reward meanings, training owns weights. Align examples with job kinds (`train.sft`, `serve.benchmark`, …). |
| [evaluation-and-environments.md](./evaluation-and-environments.md) | **rewrite** | **Keep:** Verifiers substrate; published env packages; general vs domain. **Change:** evaluation **plan** as primitive; native **traces.jsonl** as eval authority (project aggregates only); YAML+schema peer to code programs; screen/qualify consume plans. Soften “implemented vertical slice” status until APIs match 05. |
| [profiles-and-model-variants.md](./profiles-and-model-variants.md) | **delete** after absorb | Content splits into: model variant + inference binding + execution target in 02/05, and catalog composition in 05. Do not keep a competing “profiles” architecture doc. Until deleted, leave **STALE** banner. |

### ADRs

| File | Disposition | Notes |
| --- | --- | --- |
| [decisions/README.md](../decisions/README.md) | **keep** (bannered) | Index only until per-ADR reconcile. |
| [0004-lifecycle-driven-mvp-platform.md](../decisions/0004-lifecycle-driven-mvp-platform.md) | **rewrite** or **supersede** | Core platform ADR; must adopt work-package ontology and config-first catalog. |
| [0005](../decisions/0005-trackio-verifiers-traces.md) / [0006](../decisions/0006-trackio-observation-model.md) | **rewrite** | Align group key and envelope with 06; keep Verifiers-trace and observation-only decisions. |
| [0008](../decisions/0008-model-conversation-contracts.md) / [0009](../decisions/0009-native-verifiers-environment-packages.md) / [0010](../decisions/0010-environment-driven-online-rl-bridge.md) / [0011](../decisions/0011-canonical-posttraining-data.md) | **keep** (light touch) | Mostly still valid technical ownership; update nouns when touched. |
| [0001](../decisions/0001-stack-trl-native.md)–[0003](../decisions/0003-backend-neutral-evaluation-data.md) | **keep** as historical | Already marked superseded; no rewrite needed. |
| [0002](../decisions/0002-staged-runs-and-evaluation.md) / [0007](../decisions/0007-trl-vllm-025-fork.md) | **keep** | Stack/ops history; ignore for product ontology. |

---

## Mapping checklist (for rewrites)

When rewriting each concern doc, verify:

1. **Ontology:** Project → Work package (@ stage) → Run of a Job
2. **Primitives seats** filled explicitly per job kind
3. **Catalog:** base + overlay; single `catalog.resolve`; `source_layer` recorded
4. **Packages:** `common`, `data`, `serve`, `eval`, `train`, `reports`, `apps/lab`; no train↔eval↔serve imports
5. **Observation:** group = `work_package_id`; metrics namespaces from 06
6. **Non-goals:** jobs do not accept/revise/reject; Trackio is not authoring

---

## Suggested rewrite order

1. `architecture.md` (overview spine)
2. `layers-and-ownership.md`
3. `training-and-inference.md` + `evaluation-and-environments.md`
4. `trackio.md` → then thin/fold `observability.md`
5. `lineage-and-metadata.md`
6. Delete or redirect `profiles-and-model-variants.md`
7. Supersede/rewrite ADR 0004; patch 0005/0006

No implementation changes in this pass.
