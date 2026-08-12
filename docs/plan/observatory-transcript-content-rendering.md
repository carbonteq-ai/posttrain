# Render Observatory transcripts according to retained content

This ExecPlan is a living document and follows `docs/templates/PLAN.md`. The
sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes &
Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

An Observatory user opening an individual trace should see each retained
message in the form that makes its evidence understandable: prose and model
responses as safe Markdown, JSON and genuine YAML collections as navigable
structured values, code fences with syntax highlighting, tool calls and tool
results as linked structured evidence, and typed content-part arrays as
separate parts rather than one serialized blob. The exact retained value must
remain available through a raw view because rendering is a presentation over
evidence, not a replacement for it.

The behavior is accepted against two real retained populations available from
the local Observatory at `http://127.0.0.1:7861`: the Gemma 4 E2B on-policy
distillation run `opdq-ceil03-iwopd-e2b12b-c12-lb12-pb3-rseq12`, and the Qwen
3.5 4B AutomationBench evaluation run
`observatory-qwen4b-automationbench-simple-full-c16-20260804`. Tests use small
synthetic fixtures derived from their message shapes; complete production
transcripts are read-only validation evidence and are not copied into source.

This work does not change the frozen post-training product baseline. It
implements the existing requirement that Observatory present retained trace
evidence faithfully and read-only.

## Progress

- [x] (2026-08-12) Reproduced plain-text rendering in the real Gemma OPD trace
  and traced it to the shared frontend transcript component.
- [x] (2026-08-12) Inventoried 61 messages from the selected Gemma run and 120
  messages from 16 representative Qwen AutomationBench traces.
- [x] (2026-08-12) Researched maintained React Markdown, GFM, YAML, and syntax
  highlighting libraries and recorded a conservative classifier design.
- [x] (2026-08-12) Added a 25-case focused regression matrix covering real
  Gemma and Qwen shapes plus provider-neutral classification, malformed input,
  unknown roles, both tool-call conventions, typed parts, and bounded payloads.
- [x] (2026-08-12) Implemented content classification, safe Markdown,
  structured YAML/JSON, typed content parts, code highlighting, lazy raw
  disclosure, legacy function calls, and nested provider-message normalization.
- [x] (2026-08-12) Passed 71 frontend tests, TypeScript checking, the production
  build, Observatory Ruff checks, dependency audit for production packages, and
  diff validation.
- [x] (2026-08-12) Reloaded local Observatory and verified the real Gemma trace
  in the browser. Re-read the archived Qwen run through the live detail API and
  validated its exact five-message tool-exchange shape through the regression.
- [x] (2026-08-12) Recorded final evidence and kept release and deployment out
  of scope; this remains a local Observatory implementation.

## Surprises & Discoveries

- Observation: The highlighted Gemma system message is a genuine YAML mapping,
  not Markdown. Its nested mappings and sequences arrive intact, while the
  frontend places the entire value in one whitespace-preserving paragraph.
  Evidence: 19 system, 21 user, and 21 assistant messages across the retained
  run all have string content and all take the same plain paragraph path.
- Observation: Qwen tool-use traces exercise materially different message
  shapes from the Gemma run. Across 16 retained traces there are 16 system, 16
  user, 52 assistant, and 36 tool messages. Thirty-six assistant messages have
  null content plus `reasoning_content` and `tool_calls`; 36 tool messages carry
  JSON strings; 16 terminal assistant messages carry prose plus reasoning.
  Evidence: local Observatory detail reads over the named Qwen run.
- Observation: Qwen's retained tool calls use the direct Verifiers shape
  `{id, name, arguments}` rather than only the OpenAI nested
  `{id, type, function}` shape. The current `ToolCalls` component already
  accepts both.
- Observation: The server flattens `role`, `content`, `reasoning_content`, and
  `tool_calls` from a Verifiers node, but not `name`, `tool_call_id`, or legacy
  `function_call`. Those values remain under the retained nested `message`.
  The frontend therefore needs a small normalization seam rather than assuming
  every provider already returned one top-level shape.
- Observation: A YAML parser cannot be used as a generic truth test because
  valid prose is also a valid YAML scalar. YAML classification must require an
  error-free mapping or sequence plus conservative lexical evidence.

## Decision Log

- Decision: Use a deterministic content classifier rather than asking one
  heuristic library to identify arbitrary text.
  Rationale: Native arrays and objects, JSON objects/arrays, and typed content
  parts are authoritative shapes. Markdown accepts ordinary prose, while YAML
  accepts scalar strings, so Markdown is the safe textual fallback and YAML is
  accepted only for clear collection-shaped documents.
- Observation: A bounded census of the current local index covered 100 runs,
  five sources, six job kinds, and eight distinct source/job-kind groups. Four
  groups currently expose traces; their retained messages use nested system,
  user, and assistant strings plus assistant tool requests. The implementation
  contains no model, task, project, or work-package names.
- Observation: The named Qwen run remains directly readable by immutable run id
  and its representative trace still has system, user, assistant tool request,
  tool result, and assistant continuation messages. It has aged out of the
  current 100-run shell index, so the browser cannot select it through the
  sidebar even though the detail API still serves it.
- Observation: A collapsed HTML `details` element still mounts its children.
  Rendering every exact Raw payload eagerly would duplicate large transcripts
  in the initial DOM. Raw payloads now mount only after the disclosure opens;
  the real 13-message Gemma inspector initially mounts zero Raw payloads.
- Observation: Markdown, YAML, and JSON syntax-tree dependencies materially
  increase the transcript renderer chunk. The transcript is therefore loaded
  on demand. The production main chunk is 489.71 kB minified / 139.01 kB gzip,
  below the measured baseline of 503.07 kB / 142.60 kB; the transcript detail
  chunk is 437.23 kB / 134.70 kB and is paid only when evidence is opened.
  Date/Author: 2026-08-12 / Codex
- Decision: Use `react-markdown@10.1.0`, `remark-gfm@4.0.1`,
  `remark-breaks@4.0.0`, `yaml@2.9.0`, and `rehype-highlight@7.0.2`.
  Rationale: These packages compose through syntax trees, support the display
  forms present in retained model messages, and avoid raw HTML insertion.
  Syntax highlighting remains limited to fenced code blocks; it does not guess
  that an entire prompt is source code.
  Date/Author: 2026-08-12 / Codex
- Decision: Keep raw HTML disabled and retain an explicit Raw disclosure for
  every non-empty content value.
  Rationale: Trace content is untrusted evidence. Safe derived rendering must
  not execute HTML, and users must be able to compare the presentation with the
  exact retained value.
  Date/Author: 2026-08-12 / Codex
- Decision: Do not commit complete production trace contents as fixtures.
  Rationale: Tests need provider shapes and representative formatting, not
  potentially sensitive task payloads. Small synthetic fixtures preserve the
  schema and edge cases while real runs remain live acceptance evidence.
  Date/Author: 2026-08-12 / Codex
- Decision: Classification and normalization are driven only by retained value
  shape and provider-neutral message fields.
  Rationale: Gemma and Qwen are qualification populations, not product
  branches. The renderer must behave the same for any model or task that emits
  the same JSON-compatible message contract.
  Date/Author: 2026-08-12 / Codex
- Decision: Bound parsing and DOM materialization independently.
  Rationale: Native structures render at most 100 entries and eight levels,
  strings above 250,000 characters skip syntax parsers, exact Raw payloads are
  lazy, and the whole renderer is code-split. These guards preserve access to
  retained evidence without making a large trace an initial-load penalty.
  Date/Author: 2026-08-12 / Codex

## Outcomes & Retrospective

Implementation and local acceptance are complete. Observatory now renders
content according to its retained shape, with exact evidence available lazily,
and no model- or task-specific branches. The focused matrix has 25 cases and
the complete frontend suite has 71 passing tests. The production build, type
check, Ruff check, production dependency audit, and diff check pass.

The real Gemma trace exposes structured YAML and JSON fields in the browser,
retains 13 Raw disclosures, and mounts none of their payloads until opened. The
archived Qwen detail remains live and confirms the five-message tool exchange
used by the regression, but is not selectable from the current bounded run
index. No application console errors were observed. Trackio publication is not
required because it already retains the necessary evidence. Observatory has
not been released or deployed by this plan.

## Context and Orientation

`apps/observatory/src/posttrain_observatory/traces.py` projects provider trace
records into a `TraceDetail`. For Verifiers `nodes`, `project_trace` retains the
node and flattens selected fields from its nested message. The HTTP response is
provider-neutral and must continue to preserve JSON-compatible evidence.

`apps/observatory/frontend/src/components/TranscriptMessage.tsx` owns every
trace message card. `MessageBody` currently converts non-string content with
`JSON.stringify`, renders reasoning and ordinary content in plain paragraphs,
and uses `StructuredValue` only for tool results and tool-call arguments.
`Transcript` groups an assistant tool request, its tool results, and the
assistant continuation into one visual exchange.

`apps/observatory/frontend/src/components/TranscriptMessage.test.tsx` contains
the focused component tests. `apps/observatory/frontend/src/styles.css` owns
global component styling. `apps/observatory/frontend/package.json` and
`package-lock.json` own the JavaScript dependency graph.

A content part is one item in an array-valued message `content`, normally with
a discriminator such as `type: "text"`, `type: "image_url"`, or an unknown
provider-specific type. A raw view is a collapsed disclosure that renders the
original string or formatted JSON without interpreting it.

## Plan of Work

First, expand `TranscriptMessage.test.tsx` with a behavior matrix derived from
the two real populations. The Gemma cases cover YAML system prompts, Markdown
user content, structured assistant JSON, ordinary prose, and raw fidelity. The
Qwen cases cover reasoning plus a null-content tool request, direct and nested
tool-call shapes, JSON and textual tool results, an assistant continuation,
legacy `function_call`, and typed content-part arrays. Add hostile HTML and
unsafe-link cases so the safe boundary is executable.

Second, add `ContentRenderer.tsx` beside `TranscriptMessage.tsx`. It will expose
a small classifier that treats native objects and arrays as structured values,
parses JSON only when a string begins with an object or array delimiter, parses
YAML only when conservative collection markers are present and `parseDocument`
returns an error-free map or sequence, and sends all remaining text through
safe Markdown. Markdown uses GFM and hard line breaks, custom Observatory-sized
elements, raw HTML disabled, and a URL transform that permits only safe web,
mail, and local fragment links. Fenced code uses `rehype-highlight`; unlabeled
blocks remain plaintext.

Third, refactor `TranscriptMessage.tsx` around one normalization function that
reads fields from the top-level projected entry and falls back to its nested
`message`. Keep the existing assistant/tool grouping. Render reasoning through
the shared textual renderer, render modern `tool_calls` and legacy
`function_call`, render tool JSON through `StructuredValue`, and pass all other
content through the classifier. Each content body gains a collapsed Raw view.
Unknown roles and content parts remain visible through safe fallbacks.

Fourth, add tightly scoped transcript styles to `styles.css`. Headings must be
subordinate to the trace-card heading; lists, tables, blockquotes, links,
inline code, and code blocks must fit the narrow inspector without horizontal
page overflow. Structured and raw surfaces must wrap or scroll internally.

Finally, validate from the frontend outward, reload the already-running local
Observatory, and inspect the named Gemma and Qwen runs. The Gemma system prompt
must display nested structure rather than one paragraph, and its assistant JSON
must be navigable. The Qwen trace must show reasoning, tool name and arguments,
structured tool output, and final Markdown prose as one exchange. Raw values
must remain accessible in both.

## Concrete Steps

From `/home/hammad/projects/rl/apps/observatory/frontend`, add the five exact
dependencies with npm so both `package.json` and `package-lock.json` update:

    npm install react-markdown@10.1.0 remark-gfm@4.0.1 remark-breaks@4.0.0 yaml@2.9.0 rehype-highlight@7.0.2

Run the focused test while developing:

    npm test -- --run src/components/TranscriptMessage.test.tsx

Then run the complete frontend validation from the same directory:

    npm test -- --run
    npm run check
    npm run build

From `/home/hammad/projects/rl`, finish with:

    uv run ruff check apps/observatory
    git diff --check

The local server is already listening at `http://127.0.0.1:7861`. After a
production build or source change, reload the browser page before inspecting
the named runs.

## Validation and Acceptance

Focused tests must prove classification and presentation, not implementation
details. A YAML mapping must expose its field labels while its Raw disclosure
contains the exact source. Plain Markdown must produce semantic headings,
lists, links, tables, inline code, and fenced code. JSON from an assistant and
from a tool must use the structured viewer. Invalid JSON or YAML must fall back
to Markdown without throwing. A scalar such as `Done: successfully` must not be
misclassified as YAML. Raw HTML and `javascript:` links must not become active
DOM. Oversized nested evidence must remain bounded by the inspector.

The Qwen fixture must render a single assistant exchange containing reasoning,
a direct-shape tool call, a structured JSON tool result, and a Markdown final
answer. The Gemma fixture must render YAML system content and JSON assistant
content. A content-part array must render text parts and preserve unknown parts
as structured evidence. A legacy function call must remain visible.

The complete frontend suite and production build must pass. In the local
browser, a real Gemma OPD trace and a real Qwen AutomationBench trace must show
the same content categories as their tests, with no console errors and no loss
of the current trace table or tool-exchange grouping.

## Idempotence and Recovery

Dependency installation, tests, and builds are repeatable. Content detection
is pure and does not mutate provider records. If YAML classification proves too
permissive, tighten the lexical gate while keeping Markdown as the lossless
fallback. If syntax highlighting materially increases initial load time, move
it behind a lazy component without changing the classifier or acceptance
fixtures. If local source changes are not visible, rebuild the frontend and
reload the current browser tab; do not redeploy Observatory during this plan.

## Artifacts and Notes

The real evidence inventory collected on 2026-08-12 is:

    Gemma OPD: 61 messages across 12 traces
      system/string: 19
      user/string: 21
      assistant/string: 21

    Qwen AutomationBench sample: 120 messages across 16 traces
      system/string: 16
      user/string: 16
      assistant/null + reasoning + tool_calls: 36
      tool/string JSON: 36
      assistant/string + reasoning: 16

One representative Qwen branch is system, user, assistant tool request, tool
result, assistant continuation. The tool call uses direct `id`, `name`, and
`arguments` fields. The tool node retains `name` and `tool_call_id` in its
nested message.

## Interfaces and Dependencies

`ContentRenderer.tsx` should export a `MessageContent` React component and may
export its pure classifier for tests. Its props accept an unknown retained
value and an optional compact style variant. Classification returns one of
`empty`, `markdown`, `json`, `yaml`, `structured`, or `parts`, plus the parsed
value only when parsing was authoritative.

`TranscriptMessage.tsx` continues to export `StructuredValue`,
`TranscriptMessage`, and `Transcript`. The new normalization helper must not
change the HTTP schema. It resolves `role`, `content`, `reasoning_content`,
`tool_calls`, `function_call`, `name`, and `tool_call_id` across flattened and
nested message shapes.

`react-markdown`, `remark-gfm`, and `remark-breaks` own textual syntax;
`yaml` owns YAML parsing; `rehype-highlight` owns fenced-code highlighting.
Neither raw HTML parsing nor MDX execution is permitted.

Revision note (2026-08-12): Created after live Gemma and Qwen trace inventory,
internet-backed dependency research, and confirmation that the existing
failure is an Observatory presentation gap rather than missing Trackio data.
Updated after implementation with the provider-neutral test matrix, live-index
census, bundle measurements, lazy Raw behavior, and final validation results.
