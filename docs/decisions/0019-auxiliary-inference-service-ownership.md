# ADR 0019 — Auxiliary inference is an independently owned service dependency

## Status

Accepted for implementation, 2026-09-07. Remote service-handle qualification
remains open. Related plan: `docs/plan/gdpo-capo-dual-backend-support.md`.

## Context

Episode-level GDPO and CAPO can ask an environment-owned judge to score policy
trajectories. The algorithm consumes validated reward evidence; it does not own
the judge model. A local composition may launch a small judge beside training,
but a 12B judge and a 2.6B training/rollout stack do not have a safe implicit
reservation merely because both inference bindings name the same 96 GiB target.
`serve.launch` starts a child process inside the job's existing allocation; it
does not provision the judge binding's target or isolate its GPU memory.

Treating that child process as remotely managed would make a portable package
appear safe while relying on unbudgeted colocation. It would also couple judge
availability to optimizer lifecycle and make service reuse, independent scaling,
failure attribution, and cleanup ambiguous.

## Decision

- Production remote judged training uses a separately provisioned inference
  service and attaches to it. The service owner reserves its target, starts the
  selected model/runtime, authenticates the endpoint, proves readiness, retains
  immutable model and engine provenance, and stops only the service it owns.
- A training job receives a named attached-service connection plus the expected
  `InferenceBinding`. The volatile address and scoped credential reference are
  composition inputs, not catalog primitives and never algorithm settings.
- Attachment must verify deployment provenance against the selected model
  artifact/revision, renderer, backend, and relevant engine identity. `/models`
  alone proves only an alias and is insufficient for reproducible qualification.
- Environment judge plugins continue to own prompts, rubrics, schemas, retries,
  parsing, and named reward annotations. GDPO, CAPO, SAMPO, TRL, and veRL see only
  the existing validated reward-evidence contract.
- Managed child-process inference remains valid for explicitly qualified local
  or partitioned compositions. Remote colocation is rejected unless the host
  supplies an explicit resource partition and lifecycle controller covering all
  resident models; independent `gpu_memory_utilization` fractions are not such a
  reservation.
- Service readiness, saturation, timeout, and termination remain distinct from
  scorer-output validity. A failed endpoint never becomes a numeric zero reward.
- The host/work-package orchestrator owns dependency ordering: service ready,
  training admitted, training drained, then owned service release. Attached
  services are never stopped by the consumer. Interrupted training may reattach
  only to the same qualified deployment identity or must create a fresh run.

## Consequences

The initial Gemma 4 12B comparison needs an additional service deployment and
attachment gate before GDPO can run. This may consume a second GPU, but it avoids
an unmeasured OOM risk and lets judge concurrency scale independently from policy
rollouts. Vanilla GRPO and OLMo3 remain single-allocation jobs and do not wait for
or pay for a judge.

Portable packaging still contains the expected judge inference selection and
task-owned rubric. It does not contain a workstation path or a fixed ephemeral
URL. Evidence must connect the training run to the exact service deployment and
retain endpoint latency/error observations without credentials.

## Alternatives Considered

### Launch the judge in the training process allocation

Rejected for the 12B comparison because the target declaration does not reserve
a second device or coordinate vLLM sleep/wake phases. Previous scalar GRPO
evidence already reached the 96 GiB limit during actor log-probability work.

### Put model launch logic in the Verifiers judge plugin

Rejected because plugins own scoring semantics, not GPU placement, credentials,
or process lifecycle. It would also make the same rubric behave differently by
trainer backend.

### Put judge model and decoding fields in GDPO/CAPO settings

Rejected because those are inference/composition concerns and would prevent
multiple rubrics or algorithms from sharing one service safely.

### Trust a caller-supplied URL and model alias

Rejected for qualification because it cannot prove which weights or engine
served the request. It remains useful only for explicitly non-reproducible
exploration labeled as such.

## Implementation Notes

Extend the existing `AttachedInferenceService` composition path rather than
adding a scorer provider to `posttrain.train`. Add a provider-owned service
deployment receipt/handle and an attachment resolver in `posttrain.jobs` or the
Lab host. Keep execution-provider types out of reusable capability packages.
Tests must cover distinct policy/judge artifacts, credential redaction, endpoint
identity mismatch, readiness failure, consumer cleanup that leaves attached
services running, and interrupted reattachment. The live GDPO comparison remains
blocked until service placement and teardown are observed on RunPod.

## Revision History

- 2026-09-07: Accepted independent ownership and attached-service composition
  for production remote judged training.
