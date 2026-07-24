# Security policy

## Supported versions

Posttrain is pre-1.0. Security fixes are made on `main` and released for the
latest published release candidate. Older release candidates are not maintained
after a replacement is available.

The policy covers first-party `posttrain-*` packages, the CLI, Observatory, and
release automation in this repository. Vulnerabilities in maintained CarbonTeq
forks should be reported against their own repository when the affected code is
there; reports may still begin here when the boundary is unclear.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/carbonteq-ai/posttrain/security/advisories/new)
and include:

- The affected package, command, API, or deployment surface.
- The release tag or commit.
- Reproduction steps or a minimal proof of concept.
- Expected impact, including whether credentials, artifacts, traces, or remote
  execution are involved.
- Any known mitigation.

Never include production credentials, customer data, model artifacts, or
private traces in the report. Use synthetic evidence and coordinate a secure
transfer if additional material is required.

The maintainers target an acknowledgement within three business days and an
initial severity assessment within seven business days. These are response
targets rather than guaranteed service levels.

## Disclosure

Please allow maintainers time to reproduce, fix, validate, and release a
correction before public disclosure. A security release will credit the
reporter when requested and when doing so does not expose sensitive details.

## Deployment responsibility

Observatory must not be exposed on a non-loopback production address without
its configured authentication boundary. Provider credentials belong in a
secret manager or runtime environment, never in `.posttrain/`, work packages,
catalog files, logs, traces, or issue reports.
