# Product design

This directory holds revision-aware product-design references and plans. These
documents support implementation, but they do not override the frozen
[post-training product baseline](../post-training/README.md), architecture
decisions, or executable telemetry contracts.

| Product | Artifact | Status |
| --- | --- | --- |
| Observatory | [Mood board](./observatory/moodboard/README.md) | Exploratory reference; not a design contract |

Design work should distinguish among:

- **source references**, which show useful patterns from another product;
- **exploratory concepts**, which test possible Observatory layouts; and
- **accepted design contracts**, which constrain implementation after review.

Unless a product document says otherwise, Observatory is desktop-first. Mobile
may provide run status, alerts, and compact summaries, but dense chart, trace,
and comparison workflows remain desktop surfaces.
