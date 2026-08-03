# Catalog distribution

The catalog is split into a framework base layer and optional project overlay
layers. Each layer is a manifest-controlled directory of YAML documents.

```text
packages/catalog/src/posttrain/catalog/
  base/                         # packaged framework release
    layer.yaml
    models.yaml
    targets.yaml
    inference.yaml
    workloads.yaml
    environments.yaml
    evaluations.yaml
.posttrain/
  catalog/
    <overlay-id>/
      layer.yaml
      *.yaml
```

`layer.yaml` fixes the layer identity and lists every source loaded from that
directory. Schema version 1 uses the `files` list shown above. Schema version
2 uses an ordered `sources` list and can mix YAML documents with an explicit
Python provider:

```yaml
schema_version: 2
layer_id: support-agent-v1
sources:
  - kind: yaml
    path: datasets.yaml
  - kind: python
    provider: support_agent.catalog:entries
```

The provider is imported and called only while the layer is loaded. It must
return `posttrain.catalog.CatalogEntries`, containing complete typed selection
values. It is not a global registry and must not download data or execute a
dataset builder. Installed packages do not contribute entries unless a layer
lists their provider explicitly. Files and providers are unique within one
layer, and YAML paths must remain local filenames.

Catalog documents may contain the supported selection-family keys, and
selection IDs are unique within a layer. YAML and Python entries use the same
overlay and resolution rules.

Project overlays may replace a base selection by publishing the same family and
ID, or add project-local IDs. Resolution records `source_layer` and
`overlay_id`; later overlays take precedence when more than one is supplied.

## Publishing a base entry

1. Add the selection to the appropriate file under
   `packages/catalog/src/posttrain/catalog/base/`.
2. If a new document is required, add its filename to
   `packages/catalog/src/posttrain/catalog/base/layer.yaml`.
3. Use immutable artifact revisions, source revisions, and backend versions.
4. Run `uv run pytest -q apps/lab/tests/test_catalog.py` to validate schemas,
   links, overlays, and factory registrations.
5. Run the full repository validation before merging.

## Discovering entries

Use the CLI against the composed base and project catalog layers:

```bash
posttrain catalog list --family environment
posttrain catalog show environment knowledge-mmlu-pro-cot-5shot-balanced-v1
posttrain catalog show evaluation general-capability-balanced-v1
```

`show` prints the resolved immutable environment source, activation digest, and
evaluation links without importing Verifiers or downloading benchmark data.

Environment YAML stores a stable factory key rather than a Python import path.
The lab host maps that key to a callable registry. An unknown key fails while
the catalog is opened, before a run or GPU process begins.

The reusable `posttrain.common` package intentionally does not parse YAML. The
`posttrain-catalog` distribution owns the manifest-controlled YAML loader,
explicit Python provider boundary, and versioned base resource, then passes
plain mappings and already-typed selections into the framework catalog.
Project overlays remain project source under `.posttrain/catalog/`.
