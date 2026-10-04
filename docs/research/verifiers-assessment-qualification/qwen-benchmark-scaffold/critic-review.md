# Independent eligibility and benchmark review

2026-10-03. Reviewer: reused `credit_strategy` critic. The initial 133-test
qualification is a scaffold checkpoint, not acceptance of the following gaps.
The worker is correcting them; acceptance requires new regression evidence.

1. **Independent execution approval.** Automatic proof/pool model validation
   reached original replay while treating an embedded verifier binding as its
   own approval. Source hashes establish integrity, not authorization. Model
   validators must be pure. Explicit admission must receive the independently
   accepted redesign and verifier registry before executing the fixed replay.
2. **Native fact derivation.** Reusing producer-supplied `task_evidence` can allow
   changed initial/final/view facts to remain internally consistent after their
   hashes are regenerated. Deterministic eligibility needs approved producer
   derivation/replay against the retained native episode, not self-consistency
   alone. A mutation fixture must alter those facts without changing the episode
   and still be rejected.
3. **Resume result consistency.** Native-byte validation is necessary but does
   not prove a saved completion label or summary agrees with stops/errors/usage.
   Recompute the summary/status under the declared runtime contract on resume.
   Include truncation, cancellation, deadline and foreign-prefix cases.

The critic found the fixed-driver/no-archive-execution boundary, source ZIP
membership/hash checks, post-replay rechecks, independent historical-full policy,
native subject membership, slot refill and consumed-start recovery coherent.

The efficiency adviser confirmed repeated all-pool replay occurs in model
reparsing; the dispatch loop does not revalidate the full pool per attempt.
After the approval boundary is fixed, use an explicit trusted validation session
with one replay per distinct closure and byte/source checks before reuse.
Do not optimize by caching an untrusted verified boolean.

Real Qwen serving and budget enforcement remain separate qualification gates.
