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

`layer.yaml` fixes the layer identity and lists every document loaded from that
directory. Files not listed in the manifest are ignored. Catalog documents may
contain the supported selection-family keys, and selection IDs are unique
within a layer.

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

Environment YAML stores a stable factory key rather than a Python import path.
The lab host maps that key to a callable registry. An unknown key fails while
the catalog is opened, before a run or GPU process begins.

The reusable `posttrain.common` package intentionally does not parse YAML. The
`posttrain-catalog` distribution owns the manifest-controlled YAML loader and
the versioned base resource, then passes plain mappings into the framework
catalog. Project overlays remain project source under `.posttrain/catalog/`.
