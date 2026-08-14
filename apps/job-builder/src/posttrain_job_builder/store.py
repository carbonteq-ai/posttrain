"""Single-node, content-addressed storage for admitted job contexts.

This module deliberately contains no HTTP, authentication implementation, or
BuildKit calls.  Its caller supplies an authenticated principal and authorized
project scope; a later worker consumes only a sealed record.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from posttrain.common import ContractError
from posttrain.execution_pack import (
    ContextFile,
    JobBuilderCapabilities,
    JobContextTransferPlan,
    JobContextTransferReceipt,
    JobPublicationImage,
    JobPublicationPlanRequest,
    JobPublicationState,
    staged_context_directories,
)


class JobContextStore(Protocol):
    """Durable service boundary for a sealed, content-addressed job context."""

    def plan(
        self,
        *,
        principal: str,
        project_id: str,
        request: JobPublicationPlanRequest,
    ) -> JobContextTransferPlan: ...

    def put_blob(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
        digest: str,
        content: BinaryIO,
        content_length: int,
    ) -> None: ...

    def seal(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
    ) -> JobContextTransferReceipt: ...

    def cancel(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
    ) -> StoredJobPublication: ...

    def claim_next(self) -> QueuedJobPublication | None: ...

    def materialize(self, claim: QueuedJobPublication, destination: Path) -> Path: ...

    def complete(self, claim: QueuedJobPublication, image: JobPublicationImage) -> StoredJobPublication: ...

    def fail(self, claim: QueuedJobPublication, safe_error_code: str) -> StoredJobPublication: ...


@dataclass(frozen=True, slots=True)
class StoredJobPublication:
    """Private admitted record, safe to resume after a process restart."""

    request: JobPublicationPlanRequest
    state: JobPublicationState
    uploaded_blob_digests: tuple[str, ...] = ()
    queue_sequence: int | None = None
    image: JobPublicationImage | None = None
    safe_error_code: str | None = None

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.uploaded_blob_digests))) != self.uploaded_blob_digests:
            raise ContractError("stored job publication blobs must be unique and sorted")
        if self.queue_sequence is not None and self.queue_sequence < 0:
            raise ContractError("stored job publication queue sequence cannot be negative")
        if self.state in {JobPublicationState.QUEUED, JobPublicationState.BUILDING} and self.queue_sequence is None:
            raise ContractError("queued job publication requires a queue sequence")
        if self.state in {JobPublicationState.PUBLISHED, JobPublicationState.REUSED} and self.image is None:
            raise ContractError("published job publication requires an image")
        if self.image is not None and self.image.kind_image != self.request.manifest.kind_image:
            raise ContractError("stored job publication image has the wrong kind parent")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "posttrain.job-builder-store.v2",
            "request": self.request.to_payload(),
            "state": self.state.value,
            "uploaded_blob_digests": list(self.uploaded_blob_digests),
            "queue_sequence": self.queue_sequence,
            "image": self.image.to_payload() if self.image is not None else None,
            "safe_error_code": self.safe_error_code,
        }

    @classmethod
    def from_payload(cls, payload: object) -> StoredJobPublication:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "request",
            "state",
            "uploaded_blob_digests",
            "queue_sequence",
            "image",
            "safe_error_code",
        }:
            raise ContractError("stored job publication payload is invalid")
        if payload["schema"] != "posttrain.job-builder-store.v2":
            raise ContractError("stored job publication schema is unsupported")
        state = payload["state"]
        blobs = payload["uploaded_blob_digests"]
        queue_sequence = payload["queue_sequence"]
        image = payload["image"]
        safe_error_code = payload["safe_error_code"]
        if (
            not isinstance(state, str)
            or not isinstance(blobs, list)
            or not all(isinstance(item, str) for item in blobs)
            or (queue_sequence is not None and (not isinstance(queue_sequence, int) or isinstance(queue_sequence, bool)))
            or (safe_error_code is not None and not isinstance(safe_error_code, str))
        ):
            raise ContractError("stored job publication payload has invalid field types")
        try:
            parsed_state = JobPublicationState(state)
        except ValueError as error:
            raise ContractError("stored job publication state is unsupported") from error
        parsed_image: JobPublicationImage | None = None
        if image is not None:
            if not isinstance(image, dict) or set(image) != {"image", "kind_image", "cache_hit"}:
                raise ContractError("stored job publication image is invalid")
            value = image["image"]
            kind_value = image["kind_image"]
            cache_hit = image["cache_hit"]
            if not isinstance(value, str) or not isinstance(kind_value, str) or not isinstance(cache_hit, bool):
                raise ContractError("stored job publication image has invalid field types")
            from posttrain.execution import RuntimeImageRef

            parsed_image = JobPublicationImage(RuntimeImageRef(value), RuntimeImageRef(kind_value), cache_hit)
        return cls(
            JobPublicationPlanRequest.from_payload(payload["request"]),
            parsed_state,
            tuple(blobs),
            queue_sequence,
            parsed_image,
            safe_error_code,
        )


@dataclass(frozen=True, slots=True)
class QueuedJobPublication:
    """A single-flight queue lease for one sealed publication request."""

    principal: str
    project_id: str
    publication_key: str
    record: StoredJobPublication

    def __post_init__(self) -> None:
        if self.record.state is not JobPublicationState.BUILDING:
            raise ContractError("claimed job publication must be building")


class FileSystemJobContextStore:
    """A private filesystem implementation for one developer-builder node."""

    def __init__(self, *, root: Path, capabilities: JobBuilderCapabilities) -> None:
        if not root.is_absolute():
            raise ContractError("job builder store root must be absolute")
        self._root = root
        self._capabilities = capabilities
        for directory in (self._root, self._root / "blobs" / "sha256", self._root / "requests", self._root / "receipts"):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        counter = self._root / "queue-sequence"
        if not counter.exists():
            self._write_counter(counter, 0)

    def plan(
        self,
        *,
        principal: str,
        project_id: str,
        request: JobPublicationPlanRequest,
    ) -> JobContextTransferPlan:
        scope = self._scope(principal, project_id, request.publication_key)
        with self._lock(scope):
            self._admit(request, project_id)
            record = self._load(scope)
            if record is None:
                record = StoredJobPublication(request, JobPublicationState.UPLOAD_REQUIRED)
                self._write(scope, record)
            elif record.request != request:
                return JobContextTransferPlan(
                    request.publication_key,
                    JobPublicationState.BLOCKED,
                    safe_error_code="publication-conflict",
                )
            missing = tuple(
                descriptor
                for descriptor in record.request.context.files
                if not self._blob_path(descriptor.sha256).is_file()
            )
            if record.state in {JobPublicationState.PUBLISHED, JobPublicationState.REUSED}:
                return JobContextTransferPlan(request.publication_key, record.state)
            if missing:
                return JobContextTransferPlan(request.publication_key, JobPublicationState.UPLOAD_REQUIRED, missing)
            return JobContextTransferPlan(request.publication_key, record.state)

    def put_blob(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
        digest: str,
        content: BinaryIO,
        content_length: int,
    ) -> None:
        scope = self._scope(principal, project_id, publication_key)
        with self._lock(scope):
            record = self._require(scope)
            if record.state is not JobPublicationState.UPLOAD_REQUIRED:
                raise ContractError("job publication does not accept blob uploads in its current state")
            descriptor = next((item for item in record.request.context.files if item.sha256 == digest), None)
            if descriptor is None:
                raise ContractError("job publication blob is not declared by the sealed context")
            if content_length != descriptor.size_bytes:
                raise ContractError("job publication blob content length differs from the declaration")
            destination = self._blob_path(digest)
            if destination.is_file():
                self._verify_blob(destination, descriptor)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination.parent.chmod(0o700)
                descriptor_fd, temporary_name = tempfile.mkstemp(prefix=f".{digest}.", dir=destination.parent)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor_fd, "wb") as stream:
                        observed = self._copy_limited(content, stream, descriptor.size_bytes)
                        if observed != descriptor.size_bytes:
                            raise ContractError("job publication blob content length differs from the declaration")
                        stream.flush()
                        os.fsync(stream.fileno())
                    self._verify_blob(temporary, descriptor)
                    os.chmod(temporary, 0o600)
                    try:
                        os.link(temporary, destination)
                    except FileExistsError:
                        self._verify_blob(destination, descriptor)
                finally:
                    temporary.unlink(missing_ok=True)
            uploaded = tuple(sorted({*record.uploaded_blob_digests, digest}))
            self._write(
                scope,
                StoredJobPublication(
                    record.request,
                    record.state,
                    uploaded,
                    record.queue_sequence,
                    record.image,
                    record.safe_error_code,
                ),
            )

    def seal(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
    ) -> JobContextTransferReceipt:
        scope = self._scope(principal, project_id, publication_key)
        with self._lock(scope):
            record = self._require(scope)
            if record.state not in {JobPublicationState.UPLOAD_REQUIRED, JobPublicationState.QUEUED}:
                raise ContractError("job publication cannot be sealed in its current state")
            missing = [item.path.as_posix() for item in record.request.context.files if not self._blob_path(item.sha256).is_file()]
            if missing:
                raise ContractError("job publication cannot seal while declared blobs are missing")
            for descriptor in record.request.context.files:
                self._verify_blob(self._blob_path(descriptor.sha256), descriptor)
            queued = StoredJobPublication(
                record.request,
                JobPublicationState.QUEUED,
                record.uploaded_blob_digests,
                self._next_queue_sequence(),
            )
            self._write(scope, queued)
            receipt = self._receipt(queued)
            self._write_receipt(principal, project_id, receipt)
            return receipt

    def get(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
    ) -> StoredJobPublication | None:
        return self._load(self._scope(principal, project_id, publication_key))

    def cancel(
        self,
        *,
        principal: str,
        project_id: str,
        publication_key: str,
    ) -> StoredJobPublication:
        scope = self._scope(principal, project_id, publication_key)
        with self._lock(scope):
            record = self._require(scope)
            if record.state in {JobPublicationState.PUBLISHED, JobPublicationState.REUSED}:
                return record
            cancelled = StoredJobPublication(
                record.request,
                JobPublicationState.CANCELLED,
                record.uploaded_blob_digests,
                record.queue_sequence,
                record.image,
                record.safe_error_code,
            )
            self._write(scope, cancelled)
            return cancelled

    def claim_next(self) -> QueuedJobPublication | None:
        """Claim the oldest queued request, or return ``None`` when idle."""

        candidates: list[tuple[int, Path]] = []
        requests_root = self._root / "requests"
        for record_path in requests_root.rglob("record.json"):
            if record_path.is_symlink() or not record_path.is_file():
                raise ContractError("job builder queue contains an invalid publication record")
            relative = record_path.relative_to(requests_root)
            if len(relative.parts) != 4:
                raise ContractError("job builder queue record path is invalid")
            record = self._load(record_path.parent)
            if record is not None and record.state is JobPublicationState.QUEUED:
                assert record.queue_sequence is not None
                candidates.append((record.queue_sequence, record_path.parent))
        for _, scope in sorted(candidates, key=lambda item: (item[0], item[1].as_posix())):
            with self._lock(scope):
                record = self._require(scope)
                if record.state is not JobPublicationState.QUEUED:
                    continue
                building = StoredJobPublication(
                    record.request,
                    JobPublicationState.BUILDING,
                    record.uploaded_blob_digests,
                    record.queue_sequence,
                )
                self._write(scope, building)
                principal, project_id, publication_key = scope.relative_to(requests_root).parts
                return QueuedJobPublication(principal, project_id, publication_key, building)
        return None

    def materialize(self, claim: QueuedJobPublication, destination: Path) -> Path:
        """Reconstruct a fixed-layout staged context from verified blobs."""

        if not destination.is_absolute() or destination.exists():
            raise ContractError("job builder materialization destination must be a new absolute path")
        scope = self._scope(claim.principal, claim.project_id, claim.publication_key)
        with self._lock(scope):
            record = self._require(scope)
            if record != claim.record or record.state is not JobPublicationState.BUILDING:
                raise ContractError("job builder queue lease is no longer active")
            destination.mkdir(parents=True, mode=0o700)
            try:
                for relative in staged_context_directories():
                    destination.joinpath(*relative.split("/")).mkdir(parents=True, exist_ok=True, mode=0o700)
                for descriptor in record.request.context.files:
                    source = self._blob_path(descriptor.sha256)
                    self._verify_blob(source, descriptor)
                    target = destination.joinpath(*descriptor.path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copyfile(source, target)
                    target.chmod(descriptor.mode)
                return destination
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise

    def complete(self, claim: QueuedJobPublication, image: JobPublicationImage) -> StoredJobPublication:
        scope = self._scope(claim.principal, claim.project_id, claim.publication_key)
        with self._lock(scope):
            record = self._require(scope)
            if record != claim.record or record.state is not JobPublicationState.BUILDING:
                raise ContractError("job builder queue lease is no longer active")
            completed = StoredJobPublication(
                record.request,
                JobPublicationState.PUBLISHED,
                record.uploaded_blob_digests,
                record.queue_sequence,
                image,
            )
            self._write(scope, completed)
            return completed

    def fail(self, claim: QueuedJobPublication, safe_error_code: str) -> StoredJobPublication:
        if not safe_error_code or "/" in safe_error_code or "\\" in safe_error_code:
            raise ContractError("job builder failure code is invalid")
        scope = self._scope(claim.principal, claim.project_id, claim.publication_key)
        with self._lock(scope):
            record = self._require(scope)
            if record != claim.record or record.state is not JobPublicationState.BUILDING:
                raise ContractError("job builder queue lease is no longer active")
            failed = StoredJobPublication(
                record.request,
                JobPublicationState.FAILED,
                record.uploaded_blob_digests,
                record.queue_sequence,
                None,
                safe_error_code,
            )
            self._write(scope, failed)
            return failed

    def _admit(self, request: JobPublicationPlanRequest, project_id: str) -> None:
        if request.project_id != project_id:
            raise ContractError("job builder project scope differs from the package manifest")
        capabilities = self._capabilities
        if request.release_manifest_digest not in capabilities.release_manifest_digests:
            raise ContractError("job builder does not support the requested release manifest")
        if request.build_definition_digest not in capabilities.build_definition_digests:
            raise ContractError("job builder does not support the requested build definition")
        if not set(request.publication.platforms).issubset(capabilities.platforms):
            raise ContractError("job builder does not support the requested platforms")
        if len(request.context.files) > capabilities.max_file_count or request.context.total_bytes > capabilities.max_context_bytes:
            raise ContractError("job builder context exceeds its admitted limits")
        if any(item.size_bytes > capabilities.max_blob_bytes for item in request.context.files):
            raise ContractError("job builder context contains a blob above its admitted limit")
        if not capabilities.queue_available:
            raise ContractError("job builder queue is unavailable")

    def _receipt(self, record: StoredJobPublication) -> JobContextTransferReceipt:
        descriptors = record.request.context.files
        blob_sizes = {item.sha256: item.size_bytes for item in descriptors}
        uploaded = set(record.uploaded_blob_digests)
        all_digests = set(blob_sizes)
        return JobContextTransferReceipt(
            publication_key=record.request.publication_key,
            state=record.state,
            context_manifest_digest=record.request.context.digest,
            source_context_digest=record.request.context.context_digest,
            declared_file_count=len(descriptors),
            declared_bytes=record.request.context.total_bytes,
            uploaded_blob_count=len(uploaded),
            uploaded_bytes=sum(blob_sizes[digest] for digest in uploaded),
            reused_blob_count=len(all_digests - uploaded),
            reused_bytes=sum(blob_sizes[digest] for digest in all_digests - uploaded),
        )

    def _scope(self, principal: str, project_id: str, publication_key: str) -> Path:
        for label, value in (("principal", principal), ("project", project_id)):
            if not value or "/" in value or "\\" in value or value in {".", ".."}:
                raise ContractError(f"job builder {label} scope is invalid")
        if len(publication_key) != 64 or any(character not in "0123456789abcdef" for character in publication_key):
            raise ContractError("job builder publication key is invalid")
        directory = self._root / "requests" / principal / project_id / publication_key
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        return directory

    def _next_queue_sequence(self) -> int:
        lock = self._root / "queue-sequence.lock"
        with lock.open("a+b") as stream:
            os.chmod(lock, 0o600)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                path = self._root / "queue-sequence"
                try:
                    current = int(path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError) as error:
                    raise ContractError("job builder queue sequence cannot be read") from error
                if current < 0:
                    raise ContractError("job builder queue sequence is invalid")
                self._write_counter(path, current + 1)
                return current + 1
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_counter(path: Path, value: int) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".queue-sequence.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(f"{value}\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    @contextmanager
    def _lock(scope: Path) -> Iterator[None]:
        lease = scope / ".lease"
        with lease.open("a+b") as stream:
            os.chmod(lease, 0o600)
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _blob_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ContractError("job builder blob digest is invalid")
        return self._root / "blobs" / "sha256" / digest[:2] / digest

    @staticmethod
    def _copy_limited(source: BinaryIO, destination: BinaryIO, expected_size: int) -> int:
        observed = 0
        while chunk := source.read(min(1024 * 1024, expected_size - observed + 1)):
            observed += len(chunk)
            if observed > expected_size:
                raise ContractError("job publication blob exceeds its declared size")
            destination.write(chunk)
        return observed

    @staticmethod
    def _verify_blob(path: Path, descriptor: ContextFile) -> None:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != descriptor.size_bytes:
            raise ContractError("job publication blob does not match its declaration")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != descriptor.sha256:
            raise ContractError("job publication blob digest differs from its declaration")

    def _load(self, scope: Path) -> StoredJobPublication | None:
        path = scope / "record.json"
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ContractError("job builder publication record is not a regular file")
        try:
            return StoredJobPublication.from_payload(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError("job builder publication record cannot be read") from error

    def _require(self, scope: Path) -> StoredJobPublication:
        record = self._load(scope)
        if record is None:
            raise ContractError("job builder publication does not exist")
        return record

    @staticmethod
    def _write(scope: Path, record: StoredJobPublication) -> None:
        destination = scope / "record.json"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".record.", dir=scope)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(record.to_payload(), sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_receipt(self, principal: str, project_id: str, receipt: JobContextTransferReceipt) -> None:
        destination = self._root / "receipts" / principal / project_id / f"{receipt.publication_key}.json"
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.chmod(0o700)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{receipt.publication_key}.", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt.to_payload(), sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
