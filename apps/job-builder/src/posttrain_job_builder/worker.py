"""The isolated worker that turns one sealed context into an actual-job image."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError
from posttrain.execution_pack import JobImagePublicationRequest, JobImagePublisher, JobPublicationImage

from .store import FileSystemJobContextStore, StoredJobPublication


@dataclass(frozen=True, slots=True)
class JobBuildWorker:
    """Consumes at most one durable queue record at a time.

    Deployment supplies a rootless BuildKit-backed ``publisher``. The worker
    receives only the server-installed definition bundle selected by the
    publisher; its client-facing protocol never supplies a Dockerfile or build
    option.
    """

    store: FileSystemJobContextStore
    publisher: JobImagePublisher
    staging_root: Path

    def __post_init__(self) -> None:
        if not self.staging_root.is_absolute():
            raise ContractError("job builder staging root must be absolute")

    def run_one(self) -> StoredJobPublication | None:
        """Build the oldest queued publication, retaining only a safe result."""

        claim = self.store.claim_next()
        if claim is None:
            return None
        self.staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.staging_root.chmod(0o700)
        stage = Path(tempfile.mkdtemp(prefix=".job-builder-", dir=self.staging_root))
        try:
            context = self.store.materialize(claim, stage / "context")
            result = self.publisher.publish(
                JobImagePublicationRequest(
                    manifest=claim.record.request.manifest,
                    staged_context=context,
                    publication=claim.record.request.publication,
                    allow_deferred_qualification=claim.record.request.allow_deferred_qualification,
                    source_context_digest=claim.record.request.context.context_digest,
                )
            )
            if (
                result.package_key != claim.record.request.package_key
                or result.publication_key != claim.record.request.publication_key
                or result.kind_image != claim.record.request.manifest.kind_image
            ):
                raise ContractError("job builder publisher returned an image for a different sealed publication")
            return self.store.complete(claim, JobPublicationImage(result.image, result.kind_image, result.cache_hit))
        except ContractError:
            return self.store.fail(claim, "build-contract-error")
        except Exception:
            return self.store.fail(claim, "build-failed")
        finally:
            shutil.rmtree(stage, ignore_errors=True)


__all__ = ["JobBuildWorker"]
