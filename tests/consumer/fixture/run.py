"""Read back one CLI-executed run through tracking and Observatory."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from posttrain.catalog import discover_project, open_catalog
from posttrain.common import CatalogRef, ExecutionTarget
from posttrain_observatory import ObservatoryService
from posttrain_tracking_trackio import TrackioDataSource


async def _main(run_id: str) -> None:
    layout = discover_project(Path.cwd())
    catalog = open_catalog(scope=layout.project_id, overlays=layout.catalog_overlays)
    base = catalog.resolve(CatalogRef("target", "targets/local-cuda-8gb"))
    overlay = catalog.resolve(CatalogRef("target", "targets/external-cpu"))
    if not isinstance(base.value, ExecutionTarget) or not isinstance(overlay.value, ExecutionTarget):
        raise TypeError("consumer target selections did not decode")

    source = TrackioDataSource(layout.project_id)
    detail = await source.get_run(run_id)
    view = await ObservatoryService(source).get_run_view_response(run_id)
    print(
        json.dumps(
            {
                "project_id": layout.project_id,
                "project_root": str(layout.root),
                "base_catalog_release": catalog.base_id,
                "base_source": base.source_layer,
                "base_target": base.value.id,
                "overlay_source": overlay.source_layer,
                "overlay_id": overlay.overlay_id,
                "overlay_target": overlay.value.id,
                "tracking_status": detail.summary.status,
                "tracking_metrics": list(detail.metric_names),
                "observatory_mode": view.resolved_mode,
                "observatory_run_id": view.view.run.run_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run.py RUN_ID")
    asyncio.run(_main(sys.argv[1]))
