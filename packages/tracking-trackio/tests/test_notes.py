from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trackio
from posttrain.common import ContractError
from posttrain.tracking import NotesUnavailable
from posttrain.tracking.notes import verify_note_store
from posttrain_tracking_trackio import TrackioRunNotes
from trackio import context_vars

_HAS_NOTES = callable(getattr(trackio.Api, "add_run_note", None))


def test_a_client_without_notes_reports_them_unavailable() -> None:
    store = TrackioRunNotes("project", api=SimpleNamespace(runs=lambda project: []))
    assert not store.supported
    with pytest.raises(NotesUnavailable, match="0.31.5.post14.dev28"):
        asyncio.run(store.list_notes())


def test_a_server_without_notes_reports_them_unavailable() -> None:
    class OldServer:
        def __getattr__(self, name: str) -> Any:
            def call(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(f"Space 'https://trackio.example' does not support '/{name}'.")

            return call

    store = TrackioRunNotes("project", api=OldServer())
    with pytest.raises(NotesUnavailable, match="upgrade it"):
        asyncio.run(store.note_history("note"))


def test_remote_writes_need_the_write_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACKIO_WRITE_TOKEN", raising=False)
    api = SimpleNamespace(**{name: lambda *a, **k: [] for name in ("run_notes", "run_note_history", "runs")})
    api.add_run_note = api.revise_run_note = api.delete_run_note = lambda *a, **k: {}
    store = TrackioRunNotes("project", server_url="https://trackio.example", api=api)
    assert store.supported and not store.writable
    with pytest.raises(ContractError, match="TRACKIO_WRITE_TOKEN"):
        asyncio.run(store.add_note(run_id="run-a", kind="finding", body_md="x", source="cli"))


@pytest.mark.skipif(not _HAS_NOTES, reason="the installed carbonteq-trackio predates run notes (needs dev28)")
def test_local_trackio_keeps_the_note_store_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("trackio", "trackio.sqlite_storage", "trackio.utils"):
        monkeypatch.setattr(f"{module}.TRACKIO_DIR", tmp_path)
    monkeypatch.setattr("trackio.bucket_storage.TRACKIO_DIR", tmp_path)
    monkeypatch.setattr("trackio.utils.ARTIFACTS_DIR", tmp_path / "artifacts")
    context_vars.current_run.set(None)
    context_vars.current_project.set(None)
    context_vars.current_server.set(None)
    for run_id in ("posttrain-run-a", "posttrain-run-b"):
        trackio.init(project="notes-contract", name=f"{run_id}-display", config={"run_id": run_id})
        trackio.log({"train/loss": 1.0})
        trackio.finish()
    store = TrackioRunNotes("notes-contract")
    asyncio.run(verify_note_store(store, "posttrain-run-a", "posttrain-run-b"))
    listed = asyncio.run(store.list_notes("posttrain-run-b"))
    assert [note.run_id for note in listed] == ["posttrain-run-b"]
