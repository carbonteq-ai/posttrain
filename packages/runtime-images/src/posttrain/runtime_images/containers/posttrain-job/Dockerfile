# syntax=docker/dockerfile:1.7

ARG JOB_CONTEXT_REF=job-context
ARG POSTTRAIN_KIND_IMAGE
FROM ${JOB_CONTEXT_REF} AS packaged-context

FROM ${POSTTRAIN_KIND_IMAGE} AS runtime

ARG FRAMEWORK_SOURCE_DIGEST
ARG ALLOW_DEFERRED_QUALIFICATION=0
ARG CODE_REQUIREMENTS_DIGEST
ARG JOB_KIND
ARG PACKAGE_KEY
ARG PYTHON_INDEX_URL
ARG PROJECT_CONFIG_DIGEST
ARG PROJECT_SOURCE_DIGEST
ARG RESOLVED_INPUTS_DIGEST
ARG RESOLVED_CONFIG_DIGEST
ARG RUNTIME_DEPENDENCIES_DIGEST
ARG RUNTIME_VARIANT
ARG POSTTRAIN_KIND_IMAGE

ENV POSTTRAIN_JOB_ROOT="/opt/posttrain/job" \
    POSTTRAIN_VERL_PYTHONPATH="/opt/posttrain-verl/projection"

WORKDIR ${POSTTRAIN_JOB_ROOT}

# The digest-pinned kind image owns the control environment used by framework,
# project, and environment code. A backend such as veRL may expose a separate,
# independently locked environment for its worker process, but actual-job
# layers never install framework packages into that backend environment.
RUN test -n "${VIRTUAL_ENV}" && test -x "${VIRTUAL_ENV}/bin/python"

# Bind the mutable named context to the content-derived package identity before
# any other context files enter the layer graph. A build-arg-only barrier is not
# sufficient on affected BuildKit versions: the first named-context COPY may be
# reused from another context path. Copy and verify the manifest itself first,
# then keep that verified file as the manifest installed into the final image.
RUN test -n "${PACKAGE_KEY}"
COPY --from=packaged-context /package.json context-package-${PACKAGE_KEY}.json
RUN python - "${PACKAGE_KEY}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

expected = sys.argv[1]
manifest_path = Path(f"context-package-{expected}.json")
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
observed = hashlib.sha256(
    json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
).hexdigest()
if observed != expected:
    raise SystemExit(
        "named build context returned a manifest for a different package key: "
        f"expected={expected} observed={observed}"
    )
PY

# The runtime lock is the complete, hash-locked union of external
# dependencies and selected environment wheels. It is deliberately installed
# before project/framework source, configuration, and datasets so those
# frequently changing inputs do not invalidate this expensive layer.
COPY --from=packaged-context /locks/ locks/
COPY --from=packaged-context /wheels/ wheels/
RUN --mount=type=cache,id=posttrain-job-uv,target=/root/.cache/uv,sharing=locked \
    if [ -n "${PYTHON_INDEX_URL}" ]; then export UV_INDEX_URL="${PYTHON_INDEX_URL}"; fi && \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --no-deps \
      --require-hashes \
      --requirement locks/runtime.control.requirements.txt

# A veRL capsule has two independently locked environments on one interpreter.
# The control process needs the selected environment to resolve the job and
# produce the portable bridge snapshot. The backend process needs the same exact
# wheel closure to reconstruct that snapshot inside Ray. They stay separate
# because their closures conflict, not because their interpreters differ: veRL
# requires liger-kernel>=0.7,<0.8 while the control side pins 0.8.0. The shared
# kind image already owns veRL/Torch/vLLM and the rest of its locked stack.
RUN if [ "${RUNTIME_VARIANT}" = "online-rl-verl-py313" ]; then \
      test -x "/opt/posttrain-verl/bin/python"; \
      if [ -n "${PYTHON_INDEX_URL}" ]; then export UV_INDEX_URL="${PYTHON_INDEX_URL}"; fi; \
      uv pip install \
        --python "/opt/posttrain-verl/bin/python" \
        --no-deps \
        --require-hashes \
        --requirement locks/runtime.backend.requirements.txt; \
    fi

# The global runtime lock already resolves the dependency closure. This second
# lock contains only deterministic local paths under sources/framework or
# sources/project, so installing it without dependency resolution cannot
# silently fetch or select another package version.
COPY --from=packaged-context /sources/ sources/

# Framework distributions are deliberately never installed into the backend
# environment: doing so would drag their dependencies in, and those conflict
# with veRL's. The isolated veRL process nevertheless needs the backend-neutral
# contracts, portable Verifiers bridge, agent loop, and finalizer. Project those
# three namespace packages as source instead of installing a second, misleading
# copy of the framework distribution. Project code remains control-only;
# selected external environments arrive as immutable wheels.
RUN if [ "${RUNTIME_VARIANT}" = "online-rl-verl-py313" ]; then \
      install -d "${POSTTRAIN_VERL_PYTHONPATH}/posttrain"; \
      for package in common data train; do \
        source="sources/framework/packages/${package}/src/posttrain/${package}"; \
        test -d "${source}"; \
        cp -a "${source}" "${POSTTRAIN_VERL_PYTHONPATH}/posttrain/"; \
      done; \
      PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${POSTTRAIN_VERL_PYTHONPATH}" \
        "/opt/posttrain-verl/bin/python" -s -B -c \
        "from pathlib import Path; import posttrain.common, posttrain.data, posttrain.train, posttrain.train.backends.verl.worker as worker; root=Path('${POSTTRAIN_VERL_PYTHONPATH}').resolve(); modules=(posttrain.common, posttrain.data, posttrain.train, worker); assert all(Path(module.__file__).resolve().is_relative_to(root) for module in modules)"; \
    fi
# Framework and project packages retain their normal `workspace = true`
# development declarations. Reconstruct the enclosing workspace from the
# already hash-bound code requirements so uv can build those packages without
# mutating their source metadata or treating either snapshot as a standalone
# partial checkout.
RUN python - <<'PY'
import json
import tomllib
from pathlib import Path

requirements = Path("locks/code.requirements.txt").read_text(encoding="utf-8").splitlines()
members = []
for raw in requirements:
    requirement = raw.strip()
    if not requirement or requirement.startswith("#"):
        continue
    if requirement.startswith("./wheels/framework/"):
        if not Path(requirement.removeprefix("./")).is_file():
            raise SystemExit(f"staged framework wheel is missing: {requirement!r}")
        continue
    if not requirement.startswith("./sources/") or any(
        character.isspace() for character in requirement
    ):
        raise SystemExit(f"invalid code workspace member: {requirement!r}")
    member = requirement.removeprefix("./")
    if not (Path(member) / "pyproject.toml").is_file():
        raise SystemExit(f"code workspace member has no pyproject.toml: {member}")
    members.append(member)
if not members or len(members) != len(set(members)):
    raise SystemExit("code workspace members must be non-empty and unique")
# The framework contributes its uv index configuration only when packed as
# source. Installed distributions carry no workspace and need none.
framework_manifest = Path("sources/framework/pyproject.toml")
document = "[tool.uv.workspace]\nmembers = [\n"
document += "".join(f'  "{member}",\n' for member in sorted(members))
document += "]\n"
framework = (
    tomllib.loads(framework_manifest.read_text(encoding="utf-8"))
    if framework_manifest.is_file()
    else {}
)
indexes = framework.get("tool", {}).get("uv", {}).get("index", [])
if not isinstance(indexes, list):
    raise SystemExit("framework uv indexes must be an array of tables")
for index in indexes:
    if not isinstance(index, dict):
        raise SystemExit("framework uv index must be a table")
    name = index.get("name")
    url = index.get("url")
    explicit = index.get("explicit", False)
    if not isinstance(name, str) or not isinstance(url, str) or not isinstance(explicit, bool):
        raise SystemExit("framework uv index has invalid name, URL, or explicit flag")
    document += "\n[[tool.uv.index]]\n"
    document += f"name = {json.dumps(name)}\n"
    document += f"url = {json.dumps(url)}\n"
    document += f"explicit = {str(explicit).lower()}\n"
Path("pyproject.toml").write_text(document, encoding="utf-8")
PY
RUN --mount=type=cache,id=posttrain-job-uv,target=/root/.cache/uv,sharing=locked \
    uv pip install \
      --python "${VIRTUAL_ENV}" \
      --no-build-isolation \
      --no-deps \
      --no-sources \
      --reinstall \
      --requirement locks/code.requirements.txt

# Framework code is deliberately installed without dependency resolution: the
# selected kind image owns that external closure. Prove the shared runtime and
# default tracking adapter are importable here so a malformed kind fails while
# packing, before it consumes a GPU allocation.
RUN "${VIRTUAL_ENV}/bin/python" -c \
    "import posttrain_runtime, posttrain_tracking_trackio.adapter, trackio"

COPY --from=packaged-context /config/ config/
COPY --from=packaged-context /datasets/ datasets/
RUN cp "context-package-${PACKAGE_KEY}.json" package.json

# Fail the build if the staged manifest is not the authoritative contract, if
# its package key differs from the requested publication, or if it records a
# kind image other than the exact parent used by this build.
RUN test -n "${PACKAGE_KEY}" && \
    test -n "${JOB_KIND}" && \
    test -n "${FRAMEWORK_SOURCE_DIGEST}" && \
    test -n "${PROJECT_CONFIG_DIGEST}" && \
    test -n "${PROJECT_SOURCE_DIGEST}" && \
    test -n "${RESOLVED_INPUTS_DIGEST}" && \
    test -n "${RUNTIME_DEPENDENCIES_DIGEST}" && \
    test -n "${CODE_REQUIREMENTS_DIGEST}" && \
    test -n "${RESOLVED_CONFIG_DIGEST}" && \
    test -n "${RUNTIME_VARIANT}" && \
    python - \
      "${POSTTRAIN_KIND_IMAGE}" \
      "${PACKAGE_KEY}" \
      "${JOB_KIND}" \
      "${FRAMEWORK_SOURCE_DIGEST}" \
      "${PROJECT_CONFIG_DIGEST}" \
      "${PROJECT_SOURCE_DIGEST}" \
      "${RESOLVED_INPUTS_DIGEST}" \
      "${RUNTIME_DEPENDENCIES_DIGEST}" \
      "${CODE_REQUIREMENTS_DIGEST}" \
      "${RESOLVED_CONFIG_DIGEST}" \
      "${RUNTIME_VARIANT}" <<'PY'
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path

from posttrain.execution import JobPackageManifest

(
    kind_image,
    expected_package_key,
    job_kind,
    framework_source_digest,
    project_config_digest,
    project_source_digest,
    resolved_inputs_digest,
    runtime_dependencies_digest,
    code_requirements_digest,
    resolved_config_digest,
    runtime_variant,
) = sys.argv[1:]
if re.fullmatch(r".+@sha256:[0-9a-f]{64}", kind_image) is None:
    raise SystemExit("POSTTRAIN_KIND_IMAGE must be digest pinned")
manifest = JobPackageManifest.from_bytes(Path("package.json").read_bytes())
if manifest.kind_image.value != kind_image:
    raise SystemExit("package manifest kind image differs from the build parent")
if manifest.package_key != expected_package_key:
    raise SystemExit(
        "package manifest key differs from PACKAGE_KEY; "
        "usually a stale BuildKit cache — retry the pack, or run "
        "`docker buildx prune -af` if it persists"
    )
if manifest.job_kind != job_kind:
    raise SystemExit("job kind differs from the package manifest")
if manifest.framework_source_digest != framework_source_digest:
    raise SystemExit("framework source digest differs from the package manifest")
if manifest.project_config_digest != project_config_digest:
    raise SystemExit("project config digest differs from the package manifest")
if manifest.project_source_digest != project_source_digest:
    raise SystemExit("project source digest differs from the package manifest")
if manifest.resolved_inputs_digest != resolved_inputs_digest:
    raise SystemExit("resolved inputs digest differs from the package manifest")
if manifest.runtime_dependencies_digest != runtime_dependencies_digest:
    raise SystemExit("runtime dependencies digest differs from the package manifest")
if manifest.code_requirements_digest != code_requirements_digest:
    raise SystemExit("code requirements digest differs from the package manifest")
if manifest.resolved_config_digest != resolved_config_digest:
    raise SystemExit("resolved config digest differs from the package manifest")
if manifest.runtime_variant != runtime_variant:
    raise SystemExit("runtime variant differs from the package manifest")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if path.is_symlink():
            raise SystemExit(f"staged tree contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": file_digest(path),
                    "executable": bool(mode & 0o111),
                }
            )
        else:
            raise SystemExit(f"staged tree contains a special file: {relative}")
    if not entries:
        raise SystemExit(f"staged tree is empty: {root}")
    return semantic_digest({"entries": entries})


for path, expected, label in (
    (
        Path("locks/runtime.requirements.txt"),
        manifest.runtime_dependencies_digest,
        "runtime requirements",
    ),
    (
        Path("locks/code.requirements.txt"),
        manifest.code_requirements_digest,
        "code requirements",
    ),
    (
        Path("config/resolved.json"),
        manifest.resolved_config_digest,
        "resolved config",
    ),
):
    if file_digest(path) != expected:
        raise SystemExit(f"staged {label} differs from the package manifest")
for lock in manifest.runtime_dependency_locks:
    if file_digest(Path(lock.requirements_path)) != lock.requirements_digest:
        raise SystemExit(
            f"staged {lock.role} runtime dependency lock differs from the manifest"
        )
# Framework code arrives either as a source tree or as built distributions.
# Whichever route was used, its digest must be the one the manifest records.
framework_wheels = sorted(
    path for path in Path("wheels/framework").glob("*.whl") if path.is_file()
)
framework_source_files = [
    path for path in Path("sources/framework").rglob("*") if path.is_file()
]
if framework_wheels and framework_source_files:
    raise SystemExit("framework code was staged both as source and as wheels")
if framework_wheels:
    observed_framework = semantic_digest(
        [{"name": path.name, "sha256": file_digest(path)} for path in framework_wheels]
    )
elif framework_source_files:
    observed_framework = tree_digest(Path("sources/framework"))
else:
    raise SystemExit("no framework code was staged")
if observed_framework != manifest.framework_source_digest:
    raise SystemExit("staged framework code differs from the package manifest")
if tree_digest(Path("sources/project")) != manifest.project_source_digest:
    raise SystemExit("staged project source differs from the package manifest")

resolved = json.loads(Path("config/resolved.json").read_text())
expected_identities = {
    "project_id": manifest.project_id,
    "work_package_id": manifest.work_package_id,
    "job_id": manifest.job_id,
    "job_definition_id": manifest.job_definition_id,
}
if resolved.get("schema") != "posttrain.resolved-job.v1":
    raise SystemExit("resolved job schema is unsupported")
if any(resolved.get(key) != value for key, value in expected_identities.items()):
    raise SystemExit("resolved job identities differ from the package manifest")
resolved_inputs = resolved.get("resolved_inputs")
if not isinstance(resolved_inputs, dict):
    raise SystemExit("resolved job inputs must be an object")
if semantic_digest(resolved_inputs) != manifest.resolved_inputs_digest:
    raise SystemExit("resolved job inputs differ from the package manifest")

project_manifest = resolved.get("project_manifest")
selected_work_package = resolved.get("selected_work_package")
if not (
    isinstance(project_manifest, str)
    and project_manifest.startswith("project/")
    and isinstance(selected_work_package, str)
    and selected_work_package.startswith("project/")
):
    raise SystemExit("resolved project paths must stay below config/project")
project_root = Path("config/project")
project_files = [
    {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": file_digest(path),
        "size_bytes": path.stat().st_size,
    }
    for path in sorted(project_root.rglob("*"))
    if path.is_file()
]
project_payload = {
    "files": project_files,
    "project_manifest": project_manifest.removeprefix("project/"),
    "selected_work_package": selected_work_package.removeprefix("project/"),
}
if semantic_digest(project_payload) != manifest.project_config_digest:
    raise SystemExit("staged project config differs from the package manifest")

expected_wheels = set()
for package in manifest.environment_packages:
    wheel = Path("wheels/environments") / package.wheel_filename
    expected_wheels.add(package.wheel_filename)
    if (
        not wheel.is_file()
        or wheel.stat().st_size != package.wheel_size_bytes
        or file_digest(wheel) != package.wheel_digest
    ):
        raise SystemExit(
            f"staged environment wheel differs from its lock: {package.package}"
        )
observed_wheels = {
    path.name for path in Path("wheels/environments").iterdir() if path.is_file()
}
if observed_wheels != expected_wheels:
    raise SystemExit("staged environment wheel set differs from the manifest")

if manifest.backend_runtime is not None:
    backend = manifest.backend_runtime
    interpreter = next(
        lock.python_executable
        for lock in manifest.runtime_dependency_locks
        if lock.role == "backend"
    )
    if not Path(interpreter).is_file():
        raise SystemExit("veRL backend interpreter is absent from the kind image")
    dependency_lock = Path(backend.dependency_lock_path)
    if (
        not dependency_lock.is_file()
        or file_digest(dependency_lock) != backend.dependency_lock_digest
    ):
        raise SystemExit("veRL kind dependency lock differs from the package manifest")
    if tree_digest(Path(backend.projection_path)) != backend.projection_digest:
        raise SystemExit("veRL worker projection differs from the package manifest")
    worktree = Path(backend.working_directory)
    head = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != backend.source_revision or status:
        raise SystemExit("veRL kind worktree differs from the package manifest")

expected_dataset_files = set()
for dataset in manifest.datasets:
    data = Path(dataset.package_path)
    metadata = Path(dataset.manifest_path)
    assets = tuple((asset, Path(asset.package_path)) for asset in dataset.assets)
    expected_dataset_files.update(
        (data.as_posix(), metadata.as_posix(), *(path.as_posix() for _, path in assets))
    )
    if (
        data.parts[0] != "datasets"
        or metadata.parts[0] != "datasets"
        or not data.is_file()
        or not metadata.is_file()
        or data.stat().st_size != dataset.size_bytes
        or file_digest(data) != dataset.digest
    ):
        raise SystemExit(
            f"staged dataset differs from its lock: {dataset.seat_name}"
        )
    for asset, path in assets:
        if (
            path.parts[0] != "datasets"
            or not path.is_file()
            or path.stat().st_size != asset.size_bytes
            or file_digest(path) != asset.digest
        ):
            raise SystemExit(f"staged dataset asset differs from its lock: {asset.package_path}")
observed_dataset_files = {
    path.as_posix()
    for path in Path("datasets").rglob("*")
    if path.is_file()
}
if observed_dataset_files != expected_dataset_files:
    raise SystemExit("staged dataset file set differs from the manifest")
PY

LABEL org.opencontainers.image.description="Immutable posttrain actual-job image" \
      org.opencontainers.image.revision="${FRAMEWORK_SOURCE_DIGEST}" \
      org.opencontainers.image.source="https://github.com/carbonteq-ai/posttrain" \
      org.opencontainers.image.title="posttrain-job" \
      org.carbonteq.posttrain.framework-source-digest="${FRAMEWORK_SOURCE_DIGEST}" \
      org.carbonteq.posttrain.image-level="actual-job" \
      org.carbonteq.posttrain.job-kind="${JOB_KIND}" \
      org.carbonteq.posttrain.kind-image="${POSTTRAIN_KIND_IMAGE}" \
      org.carbonteq.posttrain.package-key="${PACKAGE_KEY}" \
      org.carbonteq.posttrain.project-config-digest="${PROJECT_CONFIG_DIGEST}" \
      org.carbonteq.posttrain.project-source-digest="${PROJECT_SOURCE_DIGEST}" \
      org.carbonteq.posttrain.resolved-inputs-digest="${RESOLVED_INPUTS_DIGEST}" \
      org.carbonteq.posttrain.runtime-dependencies-digest="${RUNTIME_DEPENDENCIES_DIGEST}" \
      org.carbonteq.posttrain.runtime-variant="${RUNTIME_VARIANT}"

ENTRYPOINT ["posttrain-runtime"]
CMD ["execute", "--manifest", "/opt/posttrain/job/package.json"]

FROM runtime AS smoke
RUN posttrain-runtime --help >/dev/null && \
    python -c "from pathlib import Path; from posttrain.execution import JobPackageManifest; JobPackageManifest.from_bytes(Path('/opt/posttrain/job/package.json').read_bytes())"
RUN --network=none if [ "${ALLOW_DEFERRED_QUALIFICATION}" = "1" ]; then \
      qualification_args="--allow-deferred"; \
    else \
      qualification_args=""; \
    fi && \
    posttrain-runtime qualify ${qualification_args} --timeout-seconds 60 --manifest /opt/posttrain/job/package.json && \
    if [ "${RUNTIME_VARIANT}" = "online-rl-verl-py313" ]; then \
      PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${POSTTRAIN_VERL_PYTHONPATH}" \
        "/opt/posttrain-verl/bin/python" -s -B -c \
        "from pathlib import Path; import posttrain.common, posttrain.data, posttrain.train, posttrain.train.backends.verl.worker as worker; root=Path('${POSTTRAIN_VERL_PYTHONPATH}').resolve(); modules=(posttrain.common, posttrain.data, posttrain.train, worker); assert all(Path(module.__file__).resolve().is_relative_to(root) for module in modules)"; \
    fi
