"""Durable provider-neutral admission for singular research experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from posttrain.common import ContractError, ExecutionTarget, StoredArtifactRef
from posttrain.tracking import ArtifactInput, RunSpec

from .contracts import (
    ExecutionMount,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionRequest,
    RuntimeImageRef,
)
from .service import ExecutionEvidenceSource, ExecutionSubmission, JobExecutionService

_SCHEMA = "posttrain.execution-admission.v2"
_TERMINAL_ARCHIVE_SCHEMA = "posttrain.execution-admission-terminal.v1"
_TERMINAL_RETENTION = 200
_STATES = frozenset(
    {
        "waiting",
        "submitting",
        "submission_failed",
        "submitted",
        "terminal_pending_evidence",
        "completed",
        "cancelled",
    }
)
_ACTIVE_MAPPING_STATES = frozenset({"submitting", "submission_failed", "submitted", "terminal_pending_evidence"})
_REQUIRES_RESERVATION = frozenset({"submitting", "submitted", "terminal_pending_evidence"})
type AdmissionState = Literal[
    "waiting",
    "submitting",
    "submission_failed",
    "submitted",
    "terminal_pending_evidence",
    "completed",
    "cancelled",
]
type ServiceFactory = Callable[[str, ExecutionEvidenceSource | None], JobExecutionService]
type ProviderBindingFactory = Callable[[str], str]
type PhysicalHostFactory = Callable[[ExecutionPlan], str | None]


@dataclass(frozen=True, slots=True)
class AdmissionEntry:
    run_id: str
    state: AdmissionState
    plan: ExecutionPlan
    evidence_source: ExecutionEvidenceSource | None
    queued_at: datetime
    position: int | None = None
    message: str | None = None
    control_store_uri: str | None = None


@dataclass(frozen=True, slots=True)
class Placement:
    """One execution placement and whatever currently occupies it.

    A placement is held from admission until the run reconciles, so a run that
    finished but has not been reconciled still occupies its machine. Without a
    way to see that, a queued run reports only its position and nothing
    explains what it is behind.
    """

    key: str
    provider: str
    holder: str | None = None
    holder_state: AdmissionState | None = None
    holder_since: datetime | None = None
    holder_message: str | None = None
    waiting: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    entry: AdmissionEntry
    submission: ExecutionSubmission | None = None


class ExecutionAdmissionService:
    """Admit one provider execution and hold the slot through evidence reconciliation."""

    def __init__(
        self,
        state_root: Path,
        service_factory: ServiceFactory,
        *,
        provider_binding_factory: ProviderBindingFactory | None = None,
        physical_host_factory: PhysicalHostFactory | None = None,
    ) -> None:
        if not state_root.is_absolute():
            raise ValueError("execution admission state root must be absolute")
        self._root = state_root.resolve() / "admission"
        self._snapshot = self._root / "queue.json"
        self._lock_path = self._root / "queue.lock"
        self._service_factory = service_factory
        self._provider_binding_factory = provider_binding_factory
        self._physical_host_factory = physical_host_factory

    def enqueue(
        self,
        plan: ExecutionPlan,
        *,
        evidence_source: ExecutionEvidenceSource | None,
        initial_service: JobExecutionService | None = None,
        control_store_uri: str | None = None,
    ) -> AdmissionResult:
        run_id = plan.request.run_spec.run_id
        with self._locked() as payload:
            removed = _prune_terminal(payload, keep=_TERMINAL_RETENTION)
            self._archive_terminal(removed)
            existing = _find(payload, run_id)
            encoded_plan = _encode_plan(plan)
            encoded_evidence = _encode_evidence(evidence_source)
            provider_binding = self._provider_binding(plan.provider)
            if control_store_uri is not None and (
                not control_store_uri.startswith("file://") or "\x00" in control_store_uri
            ):
                raise ContractError("admission control store locator must be a file URI")
            if existing is not None:
                if (
                    existing.get("plan") != encoded_plan
                    or existing.get("evidence_source") != encoded_evidence
                    or existing.get("provider_binding") != provider_binding
                    or existing.get("control_store_uri") != control_store_uri
                ):
                    raise ContractError(f"admission run {run_id} already names a different execution")
            else:
                payload["entries"].append(
                    {
                        "run_id": run_id,
                        "admission_key": self._admission_key(plan),
                        "state": "waiting",
                        "queued_at": datetime.now(UTC).isoformat(),
                        "plan": encoded_plan,
                        "evidence_source": encoded_evidence,
                        "provider_binding": provider_binding,
                        "control_store_uri": control_store_uri,
                    }
                )
            self._persist(payload)
        return self._pump(run_id, initial_service=initial_service)

    def status(self, run_id: str) -> tuple[AdmissionEntry, ExecutionRecord | None]:
        entry = self.get(run_id)
        if entry.state not in {
            "submitted",
            "terminal_pending_evidence",
            "completed",
        }:
            return entry, None
        service = self._service_factory(entry.plan.provider, entry.evidence_source)
        record = service.status(run_id)
        if record.state in {"succeeded", "failed", "cancelled", "lost"}:
            with self._locked() as payload:
                stored = _required(payload, run_id)
                if stored["state"] == "submitted":
                    stored["state"] = "terminal_pending_evidence"
                    self._persist(payload)
            entry = self.get(run_id)
        return entry, record

    def cancel(self, run_id: str) -> AdmissionEntry:
        entry = self.get(run_id)
        if entry.state == "waiting":
            with self._locked() as payload:
                stored = _required(payload, run_id)
                stored["state"] = "cancelled"
                stored["terminal_at"] = datetime.now(UTC).isoformat()
                self._persist(payload)
            return self.get(run_id)
        if entry.state == "submitting":
            raise ContractError("provider submission is in progress; inspect status before cancelling")
        if entry.state == "submission_failed":
            raise ContractError("provider submission outcome is unresolved; retry submission before cancelling")
        if entry.state == "submitted":
            service = self._service_factory(entry.plan.provider, entry.evidence_source)
            service.cancel(run_id)
        return self.get(run_id)

    def retry_submission(self, run_id: str) -> AdmissionResult:
        """Retry one ambiguous or failed provider submission idempotently."""

        with self._locked() as payload:
            entry = _required(payload, run_id)
            if entry["state"] not in {"submitting", "submission_failed"}:
                raise ContractError("only an unresolved provider submission can be retried")
            admission_key = str(entry["admission_key"])
            active_run_id = payload["active_by_key"].get(admission_key)
            if active_run_id not in {None, run_id}:
                raise ContractError("the execution placement is occupied by another admitted run")
            payload["active_by_key"][admission_key] = run_id
            if entry["state"] == "submission_failed":
                entry["state"] = "submitting"
                entry.pop("message", None)
                self._persist(payload)
        return self._pump(run_id, resume_submission=True)

    def acknowledge_reconciled(self, run_id: str) -> AdmissionResult | None:
        with self._locked() as payload:
            entry = _required(payload, run_id)
            if entry["state"] == "completed":
                return None
            if entry["state"] != "terminal_pending_evidence":
                raise ContractError("admission can release only after terminal provider evidence")
            entry["state"] = "completed"
            entry["terminal_at"] = datetime.now(UTC).isoformat()
            admission_key = str(entry["admission_key"])
            active_by_key = payload["active_by_key"]
            if active_by_key.get(admission_key) == run_id:
                active_by_key.pop(admission_key)
            next_entry = next(
                (
                    item
                    for item in payload["entries"]
                    if item.get("state") == "waiting" and item.get("admission_key") == admission_key
                ),
                None,
            )
            next_run_id = cast(str, next_entry["run_id"]) if next_entry is not None else None
            self._persist(payload)
        if next_run_id is None:
            return None
        try:
            return self._pump(next_run_id)
        except Exception:
            # The completed run stays released. The next run records its own
            # submission failure for explicit idempotent recovery.
            return AdmissionResult(self.get(next_run_id))

    def get(self, run_id: str) -> AdmissionEntry:
        with self._locked() as payload:
            raw = dict(_required(payload, run_id))
            entries = list(payload["entries"])
        return _decode_entry(raw, entries)

    def placements(self) -> tuple[Placement, ...]:
        """Report every known placement, who holds it, and who waits behind it."""
        payload = self._load()
        entries = list(payload["entries"])
        active = dict(payload.get("active_by_key") or {})
        keys = {str(key) for key in active}
        keys.update(str(entry["admission_key"]) for entry in entries if entry.get("state") in _ACTIVE_MAPPING_STATES)
        keys.update(str(entry["admission_key"]) for entry in entries if entry.get("state") == "waiting")

        placements: list[Placement] = []
        for key in sorted(keys):
            holder_id = active.get(key)
            holder = next(
                (entry for entry in entries if entry.get("run_id") == holder_id),
                None,
            )
            waiting = tuple(
                str(entry["run_id"])
                for entry in entries
                if entry.get("state") == "waiting" and entry.get("admission_key") == key
            )
            provider = ""
            if holder is not None:
                provider = str(_decode_plan(holder["plan"]).provider)
            elif waiting:
                first = next(entry for entry in entries if entry.get("run_id") == waiting[0])
                provider = str(_decode_plan(first["plan"]).provider)
            placements.append(
                Placement(
                    key=key,
                    provider=provider,
                    holder=str(holder_id) if holder_id else None,
                    holder_state=(cast(AdmissionState, holder["state"]) if holder is not None else None),
                    holder_since=(datetime.fromisoformat(str(holder["queued_at"])) if holder is not None else None),
                    holder_message=(
                        str(holder["message"])
                        if holder is not None and isinstance(holder.get("message"), str)
                        else None
                    ),
                    waiting=waiting,
                )
            )
        return tuple(placements)

    def list(self) -> tuple[AdmissionEntry, ...]:
        with self._locked() as payload:
            entries = [dict(item) for item in payload["entries"]]
        return tuple(_decode_entry(item, entries) for item in entries)

    def _pump(
        self,
        requested_run_id: str | None,
        *,
        initial_service: JobExecutionService | None = None,
        resume_submission: bool = False,
    ) -> AdmissionResult:
        with self._locked() as payload:
            owns_transition = False
            requested = _required(payload, requested_run_id) if requested_run_id is not None else None
            admission_key = str(requested["admission_key"]) if requested is not None else None
            active_by_key = payload["active_by_key"]
            active_id = active_by_key.get(admission_key) if admission_key is not None else None
            if active_id is None:
                waiting = next(
                    (
                        item
                        for item in payload["entries"]
                        if item.get("state") == "waiting"
                        and (admission_key is None or item.get("admission_key") == admission_key)
                    ),
                    None,
                )
                if waiting is not None:
                    waiting["state"] = "submitting"
                    owns_transition = True
                    active_id = waiting["run_id"]
                    admission_key = str(waiting["admission_key"])
                    active_by_key[admission_key] = active_id
                    self._persist(payload)
            active = _find(payload, cast(str | None, active_id))
            if active is None or active.get("state") != "submitting":
                if requested is None:
                    raise KeyError("admission run not found")
                entries = list(payload["entries"])
                return AdmissionResult(_decode_entry(dict(requested), entries))
            if not owns_transition and not resume_submission:
                if requested is None:
                    raise KeyError("admission run not found")
                entries = list(payload["entries"])
                return AdmissionResult(_decode_entry(dict(requested), entries))
            active_copy = dict(active)

        entry = _decode_entry(active_copy, [active_copy])
        with self._submission_claim(entry.run_id) as claimed:
            if not claimed:
                requested_id = requested_run_id or entry.run_id
                return AdmissionResult(self.get(requested_id))

            # Re-read after taking the per-run claim. The process that first
            # persisted ``submitting`` may have died, or another caller may
            # already have completed the idempotent provider submission.
            with self._locked() as payload:
                active = _required(payload, entry.run_id)
                admission_key = str(active["admission_key"])
                if active["state"] != "submitting" or payload["active_by_key"].get(admission_key) != entry.run_id:
                    requested = _required(
                        payload,
                        requested_run_id or entry.run_id,
                    )
                    entries = list(payload["entries"])
                    return AdmissionResult(_decode_entry(dict(requested), entries))
                active_copy = dict(active)

            entry = _decode_entry(active_copy, [active_copy])
            expected_binding = active_copy.get("provider_binding")
            current_binding = self._provider_binding(entry.plan.provider)
            if expected_binding != current_binding:
                with self._locked() as payload:
                    active = _required(payload, entry.run_id)
                    if active["state"] == "submitting":
                        active["state"] = "submission_failed"
                        active["message"] = (
                            "provider binding changed after admission; restore the original binding or create a new run"
                        )
                        # Keep the placement quarantined. This can be a retry
                        # after an earlier ambiguous provider response.
                        self._persist(payload)
                raise ContractError("execution provider binding changed after admission")
            service = initial_service or self._service_factory(
                entry.plan.provider,
                entry.evidence_source,
            )
            try:
                submission = service.submit(entry.plan)
            except Exception as error:
                with self._locked() as payload:
                    active = _required(payload, entry.run_id)
                    if active["state"] == "submitting":
                        active["state"] = "submission_failed"
                        active["message"] = (
                            f"{type(error).__name__}: {error}; provider submission "
                            "outcome is unresolved, so retry explicitly with the "
                            "same run identity once the cause is addressed"
                        )
                        # A provider can accept the deterministic run before
                        # the response is lost. Preserve the worker reservation
                        # until an idempotent retry resolves that ambiguity.
                        self._persist(payload)
                # Carry the cause into the message. Without it a deterministic
                # failure, such as an unset environment variable, reads as an
                # ambiguous provider response and the advice to retry loops
                # forever with nothing to act on.
                raise RuntimeError(
                    f"provider submission is unresolved for run {entry.run_id}: "
                    f"{type(error).__name__}: {error}. Once that is addressed, "
                    f"retry with `posttrain run retry-submit {entry.run_id}`"
                ) from error
            with self._locked() as payload:
                active = _required(payload, entry.run_id)
                if active["state"] == "submitting":
                    active["state"] = "submitted"
                    self._persist(payload)
                requested = _required(payload, requested_run_id or entry.run_id)
                entries = list(payload["entries"])
            return AdmissionResult(
                _decode_entry(dict(requested), entries),
                submission if requested_run_id in {None, entry.run_id} else None,
            )

    @contextmanager
    def _submission_claim(self, run_id: str) -> Any:
        """Own one provider call without serializing unrelated worker queues.

        Kernel file locks are released automatically if a CLI process exits or
        is killed. A later explicit retry can therefore resume the same
        deterministic submission while concurrent callers remain read-only.
        """

        digest = hashlib.sha256(run_id.encode()).hexdigest()
        path = self._root / f"submit-{digest}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        claimed = False
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            claimed = True
            yield True
        finally:
            if claimed:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @contextmanager
    def _locked(self) -> Any:
        self._root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self._lock_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            payload = self._load()
            yield payload
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._snapshot.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema": _SCHEMA, "active_by_key": {}, "entries": []}
        except json.JSONDecodeError as error:
            raise ContractError("execution admission snapshot is invalid") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _SCHEMA
            or not isinstance(payload.get("entries"), list)
            or not isinstance(payload.get("active_by_key"), dict)
        ):
            raise ContractError("execution admission snapshot schema is unsupported")
        _validate_payload(payload)
        return payload

    def _persist(self, payload: dict[str, Any]) -> None:
        _validate_payload(payload)
        temporary = self._root / f".queue-{uuid.uuid4().hex}.tmp"
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self._snapshot)
        os.chmod(self._snapshot, 0o600)
        directory = os.open(self._root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _archive_terminal(self, entries: tuple[dict[str, Any], ...]) -> None:
        """Retain compact terminal admission evidence before snapshot pruning."""

        if not entries:
            return
        root = self._root / "terminal"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for entry in entries:
            run_id = str(entry["run_id"])
            digest = hashlib.sha256(run_id.encode()).hexdigest()
            path = root / f"{digest}.json"
            receipt = {
                "schema": _TERMINAL_ARCHIVE_SCHEMA,
                "run_id": run_id,
                "state": entry["state"],
                "queued_at": entry["queued_at"],
                "terminal_at": entry.get("terminal_at"),
                "admission_key": entry["admission_key"],
                "provider": entry["plan"].get("provider"),
                "job_image": entry["plan"].get("request", {}).get("image"),
            }
            if path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise ContractError(f"terminal admission receipt is invalid for run {run_id}") from error
                if existing != receipt:
                    raise ContractError(f"terminal admission receipt conflicts for run {run_id}")
                continue
            temporary = root / f".{digest}-{uuid.uuid4().hex}.tmp"
            encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, path)
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def _provider_binding(self, provider: str) -> str:
        if self._provider_binding_factory is None:
            return provider
        binding = self._provider_binding_factory(provider)
        if not binding.strip() or "\x00" in binding:
            raise ContractError("execution provider binding identity is invalid")
        return binding

    def _admission_key(self, plan: ExecutionPlan) -> str:
        configured_host = self._physical_host_factory(plan) if self._physical_host_factory is not None else None
        return _admission_key(
            plan,
            configured_local_hostname=configured_host,
            require_configured_local_hostname=self._physical_host_factory is not None,
        )


def _find(payload: dict[str, Any], run_id: str | None) -> dict[str, Any] | None:
    if run_id is None:
        return None
    return next(
        (item for item in payload["entries"] if item.get("run_id") == run_id),
        None,
    )


SELF_SCHEDULING_PROVIDERS = frozenset({"dstack"})
"""Providers that decide placement themselves.

Admission exists to keep two runs off one machine when nothing else will. A
provider with its own scheduler already does that, and does it across every
client rather than only this one, so competing with it would be both redundant
and wrong.
"""


def _admission_key(
    plan: ExecutionPlan,
    *,
    configured_local_hostname: str | None = None,
    require_configured_local_hostname: bool = False,
) -> str:
    if plan.provider in SELF_SCHEDULING_PROVIDERS:
        # This provider places runs on its own fleet, across clients this
        # process cannot see. Holding an exclusive key here would arbitrate
        # nothing while still forcing a target to name a specific machine, so
        # each run holds only itself and nothing ever queues behind it.
        return f"run:{plan.request.run_spec.run_id}"
    instances = plan.request.target.placement.get("instances")
    if isinstance(instances, list):
        hostnames = tuple(
            str(item["hostname"])
            for item in instances
            if isinstance(item, dict) and isinstance(item.get("hostname"), str)
        )
        if len(hostnames) == 1 and len(instances) == 1:
            return "host:" + _normalized_hostname(hostnames[0])
    if plan.provider in {"local", "local-docker"}:
        if configured_local_hostname is not None:
            return "host:" + _normalized_hostname(configured_local_hostname)
        if require_configured_local_hostname:
            raise ContractError(
                "local admission requires providers.local.canonical_hostname "
                "when the execution target has no exact hostname"
            )
        return "host:local-docker"
    return f"provider:{plan.provider}:target:{plan.request.target.id}"


def _normalized_hostname(value: str) -> str:
    hostname = value.strip().lower().rstrip(".")
    if (
        not hostname
        or "\x00" in hostname
        or len(hostname) > 253
        or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or re.fullmatch(r"[a-z0-9-]+", label) is None
            for label in hostname.split(".")
        )
    ):
        raise ContractError("execution admission hostname is invalid")
    return hostname


def _required(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    entry = _find(payload, run_id)
    if entry is None:
        raise KeyError(f"admission run not found: {run_id}")
    return entry


def _validate_payload(payload: dict[str, Any]) -> None:
    entries = payload["entries"]
    active_by_key = payload["active_by_key"]
    run_ids: set[str] = set()
    by_run: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContractError("execution admission entry is invalid")
        run_id = entry.get("run_id")
        admission_key = entry.get("admission_key")
        state = entry.get("state")
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in run_ids
            or not isinstance(admission_key, str)
            or not admission_key
            or state not in _STATES
            or not isinstance(entry.get("plan"), dict)
        ):
            raise ContractError("execution admission entry identity is invalid")
        run_ids.add(run_id)
        by_run[run_id] = entry
    for admission_key, run_id in active_by_key.items():
        entry = by_run.get(run_id)
        if (
            not isinstance(admission_key, str)
            or not isinstance(run_id, str)
            or entry is None
            or entry.get("admission_key") != admission_key
            or entry.get("state") not in _ACTIVE_MAPPING_STATES
        ):
            raise ContractError("execution admission active placement is invalid")
    for entry in entries:
        if entry["state"] in _REQUIRES_RESERVATION:
            if active_by_key.get(entry["admission_key"]) != entry["run_id"]:
                raise ContractError("execution admission active entry has no placement reservation")


def _prune_terminal(
    payload: dict[str, Any],
    *,
    keep: int,
) -> tuple[dict[str, Any], ...]:
    terminal = [entry for entry in payload["entries"] if entry.get("state") in {"completed", "cancelled"}]
    if len(terminal) <= keep:
        return ()
    retained_ids = {
        str(entry["run_id"])
        for entry in sorted(
            terminal,
            key=lambda item: str(item.get("queued_at", "")),
            reverse=True,
        )[:keep]
    }
    removed = tuple(
        entry
        for entry in payload["entries"]
        if entry.get("state") in {"completed", "cancelled"} and entry.get("run_id") not in retained_ids
    )
    payload["entries"] = [
        entry
        for entry in payload["entries"]
        if entry.get("state") not in {"completed", "cancelled"} or entry.get("run_id") in retained_ids
    ]
    return removed


def _decode_entry(raw: dict[str, Any], entries: list[dict[str, Any]]) -> AdmissionEntry:
    waiting = [
        item["run_id"]
        for item in entries
        if item.get("state") == "waiting" and item.get("admission_key") == raw.get("admission_key")
    ]
    run_id = str(raw["run_id"])
    position = waiting.index(run_id) + 1 if run_id in waiting else None
    queued_at = datetime.fromisoformat(str(raw["queued_at"]))
    if queued_at.tzinfo is None:
        raise ContractError("execution admission timestamp must be timezone-aware")
    return AdmissionEntry(
        run_id=run_id,
        state=cast(AdmissionState, raw["state"]),
        plan=_decode_plan(raw["plan"]),
        evidence_source=_decode_evidence(raw.get("evidence_source")),
        queued_at=queued_at,
        position=position,
        message=(str(raw["message"]) if isinstance(raw.get("message"), str) else None),
        control_store_uri=(str(raw["control_store_uri"]) if isinstance(raw.get("control_store_uri"), str) else None),
    )


def _encode_evidence(value: ExecutionEvidenceSource | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "provider": value.provider,
        "source_id": value.source_id,
        "project": value.project,
        "endpoint": value.endpoint,
        "scope": value.scope,
    }


def _decode_evidence(value: object) -> ExecutionEvidenceSource | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError("execution admission evidence source is invalid")
    return ExecutionEvidenceSource(**value)


def _encode_plan(plan: ExecutionPlan) -> dict[str, Any]:
    request = plan.request
    spec = request.run_spec
    return {
        "provider": plan.provider,
        "native_plan_id": plan.native_plan_id,
        "details": _json_value(plan.details),
        "request": {
            "run_spec": {
                "project_id": spec.project_id,
                "work_package_id": spec.work_package_id,
                "stage": spec.stage,
                "job_kind": spec.job_kind,
                "job_definition_version": spec.job_definition_version,
                "run_id": spec.run_id,
                "resolved_inputs": _json_value(spec.resolved_inputs),
                "source_metadata": _json_value(spec.source_metadata),
                "required_artifact_roles": list(spec.required_artifact_roles),
                "artifacts": {
                    name: {
                        "kind": item.kind,
                        "reference": {
                            "provider": item.reference.provider,
                            "namespace": item.reference.namespace,
                            "name": item.reference.name,
                            "version": item.reference.version,
                            "digest": item.reference.digest,
                            "provider_metadata": _json_value(item.reference.provider_metadata),
                        },
                    }
                    for name, item in spec.artifacts.items()
                },
            },
            "job_definition_id": request.job_definition_id,
            "image": request.image.value,
            "target": {
                "id": request.target.id,
                "revision": request.target.revision,
                "device_class": request.target.device_class,
                "memory_gb": request.target.memory_gb,
                "placement": _json_value(request.target.placement),
                "host_constraints": _json_value(request.target.host_constraints),
            },
            "command": list(request.command),
            "idempotency_key": request.idempotency_key,
            "policy": {
                "timeout_seconds": request.policy.timeout_seconds,
                "max_attempts": request.policy.max_attempts,
                "priority": request.policy.priority,
            },
            "attempt": request.attempt,
            "environment_names": list(request.environment_names),
            "mounts": [
                {
                    "instance_path": str(item.instance_path),
                    "container_path": str(item.container_path),
                    "purpose": item.purpose,
                    "optional": item.optional,
                }
                for item in request.mounts
            ],
        },
    }


def _decode_plan(value: object) -> ExecutionPlan:
    if not isinstance(value, dict) or not isinstance(value.get("request"), dict):
        raise ContractError("execution admission plan is invalid")
    raw = value["request"]
    spec = raw["run_spec"]
    artifacts = {
        name: ArtifactInput(
            reference=StoredArtifactRef(**item["reference"]),
            kind=item["kind"],
        )
        for name, item in spec.get("artifacts", {}).items()
    }
    run_spec = RunSpec(
        project_id=spec["project_id"],
        work_package_id=spec["work_package_id"],
        stage=spec["stage"],
        job_kind=spec["job_kind"],
        job_definition_version=spec["job_definition_version"],
        run_id=spec["run_id"],
        resolved_inputs=spec.get("resolved_inputs", {}),
        source_metadata=spec.get("source_metadata", {}),
        artifacts=artifacts,
        required_artifact_roles=tuple(spec.get("required_artifact_roles", ())),
    )
    target = ExecutionTarget(**raw["target"])
    policy = ExecutionPolicy(**raw["policy"])
    mounts = tuple(
        ExecutionMount(
            instance_path=Path(item["instance_path"]),
            container_path=Path(item["container_path"]),
            purpose=item["purpose"],
            optional=bool(item.get("optional", False)),
        )
        for item in raw.get("mounts", ())
    )
    request = ExecutionRequest(
        run_spec=run_spec,
        job_definition_id=raw["job_definition_id"],
        image=RuntimeImageRef(raw["image"]),
        target=target,
        command=tuple(raw["command"]),
        idempotency_key=raw["idempotency_key"],
        policy=policy,
        attempt=int(raw.get("attempt", 1)),
        environment_names=tuple(raw.get("environment_names", ())),
        mounts=mounts,
    )
    return ExecutionPlan(
        provider=value["provider"],
        request=request,
        native_plan_id=value.get("native_plan_id"),
        details=dict(value.get("details", {})),
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
