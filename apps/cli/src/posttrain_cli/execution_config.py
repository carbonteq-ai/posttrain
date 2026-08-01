"""Layered, secret-free execution configuration for the primary CLI."""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import tomllib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlparse

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution import RuntimeImageRef
from posttrain.project import (
    ExecutionOverrides,
    LaunchOverrides,
    PackageOverrides,
    ResolvedExecutionSettings,
    SettingSource,
    resolve_execution_settings,
    resolve_runtime_environment,
)
from posttrain.runtime_images import cached_definition_root
from posttrain.runtime_images.manifest import (
    ManifestError,
    PublishedManifest,
    load_manifest,
)
from posttrain_execution_buildkit import KindDependencyConstraints

REGISTRY_ENVIRONMENT_VARIABLE = "POSTTRAIN_REGISTRY"
"""Names the registry a project pushes its own actual-job images to.

This is the project's registry, not the framework's. The framework publishes
its base and job-kind images to the prefix recorded in the shipped manifest;
that location is a property of the release and is never configured here.
"""

_JOB_REPOSITORY_SUFFIX = "posttrain-job"


@dataclass(frozen=True, slots=True)
class ExecutionStorageBinding:
    run_root: Path
    model_cache: Path | None = None
    compile_cache: Path | None = None


@dataclass(frozen=True, slots=True)
class LocalProviderBinding:
    canonical_hostname: str | None = None
    storage: ExecutionStorageBinding | None = None
    trust_bundle: Path | None = None


@dataclass(frozen=True, slots=True)
class DstackBinding:
    project: str
    python: Path
    environment_file: Path | None = None
    storage: ExecutionStorageBinding | None = None
    trust_bundle: Path | None = None
    capacity_wait_seconds: int = 0


@dataclass(frozen=True, slots=True)
class MachineTrackingBinding:
    """Secret-free tracking default supplied to every project on a machine."""

    kind: Literal["trackio", "wandb"]
    endpoint: str | None = None
    credentials: str | None = None


@dataclass(frozen=True, slots=True)
class MachineServicesBinding:
    """Internal service defaults shared by projects on one machine."""

    python_index_url: str | None = None
    python_index_credentials: str | None = None
    job_registry: str | None = None


@dataclass(frozen=True, slots=True)
class MachineHuggingFaceBinding:
    """Hugging Face credential selection shared by jobs on one machine."""

    credentials: str | None = None


@dataclass(frozen=True, slots=True)
class MachineConfig:
    """Operator-owned defaults loaded from the current user's config home."""

    name: str
    path: Path
    projects: tuple[Path, ...]
    defaults: ExecutionOverrides
    local: LocalProviderBinding
    dstack: DstackBinding | None
    tracking: MachineTrackingBinding | None
    huggingface: MachineHuggingFaceBinding | None
    services: MachineServicesBinding
    credentials: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class RegistryBinding:
    repository: str
    universal_image: RuntimeImageRef
    kind_images: Mapping[str, RuntimeImageRef]
    constraint_profiles: Mapping[str, ConstraintProfileBinding]
    mirror_prefix: str | None = None
    buildx_builder: str | None = None
    receipt_root: Path | None = None
    bake_file: Path | None = None
    framework_source_root: Path | None = None


@dataclass(frozen=True, slots=True)
class ConstraintProfileBinding:
    path: Path
    contents_digest: str
    provided_packages: tuple[str, ...]
    digest: str
    backend_path: Path | None = None
    backend_contents_digest: str | None = None
    backend_provided_packages: tuple[str, ...] = ()
    backend_digest: str | None = None


@dataclass(frozen=True, slots=True)
class LocalExecutionConfig:
    path: Path
    defaults: ExecutionOverrides = ExecutionOverrides()
    environment_file: Path | None = None
    local: LocalProviderBinding | None = None
    dstack: DstackBinding | None = None
    registry: RegistryBinding | None = None
    machine: MachineConfig | None = None


_DEFAULT_LOCAL_NAME = "execution.toml"
_DEFAULT_KEYS = {field.name for field in fields(ExecutionOverrides)}


def load_local_execution_config(
    layout: ProjectLayout,
    *,
    path: Path | None = None,
    env_file: Path | None = None,
) -> LocalExecutionConfig:
    """Resolve machine defaults plus one project's protected runtime values."""

    configured = (path or layout.state / _DEFAULT_LOCAL_NAME).expanduser().resolve()
    runtime_environment = resolve_runtime_environment(layout.root, env_file=env_file)
    # An explicit path is the compatibility and test escape hatch for the
    # project-local v0.2 binding. Normal project opens always use the
    # automatically discovered machine configuration.
    machine = None if path is not None else load_machine_config()
    if machine is not None:
        provisional = LocalExecutionConfig(
            path=machine.path,
            defaults=machine.defaults,
            environment_file=runtime_environment.path,
            local=machine.local,
            dstack=machine.dstack,
            machine=machine,
        )
        return LocalExecutionConfig(
            path=machine.path,
            defaults=machine.defaults,
            environment_file=runtime_environment.path,
            local=machine.local,
            dstack=machine.dstack,
            registry=derived_registry(environ=load_execution_environment(provisional)),
            machine=machine,
        )
    if not configured.exists():
        # A project with no machine binding is still fully usable: the release
        # pins every framework image, so only the project's own registry is
        # missing, and the environment can supply it.
        return LocalExecutionConfig(
            configured,
            environment_file=runtime_environment.path,
            registry=derived_registry(environ=runtime_environment.for_execution()),
        )
    if not configured.is_file():
        raise ContractError(f"execution configuration is not a file: {configured}")
    if configured.stat().st_mode & 0o077:
        raise ContractError(f"execution configuration must not be accessible by group or others: {configured}")
    try:
        payload = tomllib.loads(configured.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"invalid execution configuration {configured}: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"invalid execution configuration {configured}")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "environment_file",
            "defaults",
            "providers",
            "registry",
        },
        "root",
    )
    if payload.get("schema_version") != 1:
        raise ContractError("execution configuration schema_version must be 1")

    defaults = _parse_overrides(payload.get("defaults"), context="defaults")
    environment_value = payload.get("environment_file")
    legacy_environment_file = (
        _configured_path(
            environment_value,
            configured.parent,
            "environment_file",
        )
        if environment_value is not None
        else None
    )
    if legacy_environment_file is not None:
        _require_protected_file(legacy_environment_file, "execution environment file")
    if runtime_environment.path is not None:
        environment_file = runtime_environment.path
    else:
        environment_file = legacy_environment_file
        if environment_file is not None:
            warnings.warn(
                "execution.toml environment_file is deprecated; use project-root posttrain.env instead",
                DeprecationWarning,
                stacklevel=2,
            )
    providers = _mapping(payload.get("providers"), context="providers", allow_none=True)
    _reject_unknown(providers, {"local", "dstack"}, "providers")
    local = _parse_local(providers.get("local"), base=configured.parent)
    dstack = _parse_dstack(providers.get("dstack"), base=configured.parent)
    # A file that says nothing about the registry is not a file that says there
    # is no registry. Writing execution.toml for an unrelated setting, such as
    # the local provider's hostname, otherwise discards POSTTRAIN_REGISTRY and
    # reports the project as having nowhere to publish.
    provisional = LocalExecutionConfig(
        path=configured,
        defaults=defaults,
        environment_file=environment_file,
        local=local,
        dstack=dstack,
    )
    runtime_values = load_execution_environment(provisional)
    parsed_registry = _parse_registry(
        payload.get("registry"),
        base=configured.parent,
        environ=runtime_values,
    )
    registry = parsed_registry or derived_registry(environ=runtime_values)
    return LocalExecutionConfig(
        path=configured,
        defaults=defaults,
        environment_file=environment_file,
        local=local,
        dstack=dstack,
        registry=registry,
    )


def provider_binding_fingerprint(
    config: LocalExecutionConfig,
    provider: str,
) -> str:
    """Hash one secret-free machine binding used to construct a provider."""

    if provider == "local-docker":
        binding = config.local
        payload = {
            "provider": provider,
            "canonical_hostname": (binding.canonical_hostname if binding is not None else None),
            "trust_bundle": (
                str(binding.trust_bundle) if binding is not None and binding.trust_bundle is not None else None
            ),
            "storage": _storage_identity(binding.storage if binding is not None else None),
        }
    elif provider == "dstack":
        binding = config.dstack
        if binding is None:
            raise ContractError("dstack provider binding is unavailable")
        payload = {
            "provider": provider,
            "project": binding.project,
            "python": str(binding.python),
            # Secret values may rotate; the protected file path is stable
            # provider identity, while its contents remain launch-only.
            "environment_file": (str(binding.environment_file) if binding.environment_file is not None else None),
            "trust_bundle": (str(binding.trust_bundle) if binding.trust_bundle is not None else None),
            "storage": _storage_identity(binding.storage),
            "capacity_wait_seconds": binding.capacity_wait_seconds,
        }
    else:
        # Third-party providers retain a stable name-only identity until their
        # adapter exposes additional secret-free machine binding fields.
        payload = {"provider": provider}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _storage_identity(binding: ExecutionStorageBinding | None) -> object:
    if binding is None:
        return None
    return {
        "run_root": str(binding.run_root),
        "model_cache": (str(binding.model_cache) if binding.model_cache is not None else None),
        "compile_cache": (str(binding.compile_cache) if binding.compile_cache is not None else None),
    }


def _parse_overrides(value: object, *, context: str) -> ExecutionOverrides:
    payload = _mapping(value, context=context, allow_none=True)
    _reject_unknown(payload, _DEFAULT_KEYS, context)
    environment_names = _string_tuple(
        payload.get("environment_names"),
        context=f"{context}.environment_names",
        allow_none=True,
    )
    return ExecutionOverrides(
        provider=_optional_config_string(payload.get("provider"), f"{context}.provider"),
        target=_optional_config_string(payload.get("target"), f"{context}.target"),
        runtime_profile=_optional_config_string(
            payload.get("runtime_profile"),
            f"{context}.runtime_profile",
        ),
        timeout_seconds=_optional_positive_int(
            payload.get("timeout_seconds"),
            f"{context}.timeout_seconds",
        ),
        max_attempts=_optional_positive_int(
            payload.get("max_attempts"),
            f"{context}.max_attempts",
        ),
        priority=_optional_int(payload.get("priority"), f"{context}.priority"),
        environment_names=environment_names,
    )


def _parse_local(value: object, *, base: Path) -> LocalProviderBinding | None:
    if value is None:
        return None
    payload = _mapping(value, context="providers.local")
    _reject_unknown(
        payload,
        {"canonical_hostname", "storage", "trust_bundle"},
        "providers.local",
    )
    trust_bundle = _optional_configured_path(
        payload.get("trust_bundle"),
        base,
        "providers.local.trust_bundle",
    )
    if trust_bundle is not None and not trust_bundle.is_file():
        raise ContractError(f"local execution trust bundle is missing: {trust_bundle}")
    return LocalProviderBinding(
        canonical_hostname=_optional_hostname(
            payload.get("canonical_hostname"),
            "providers.local.canonical_hostname",
        ),
        storage=_parse_storage(
            payload.get("storage"),
            base=base,
            context="providers.local.storage",
        ),
        trust_bundle=trust_bundle,
    )


def _parse_dstack(value: object, *, base: Path) -> DstackBinding | None:
    if value is None:
        return None
    payload = _mapping(value, context="providers.dstack")
    _reject_unknown(
        payload,
        {
            "project",
            "python",
            "environment_file",
            "storage",
            "trust_bundle",
            "capacity_wait_seconds",
        },
        "providers.dstack",
    )
    project = _required_config_string(payload.get("project"), "providers.dstack.project")
    python = _configured_executable_path(
        payload.get("python"),
        base,
        "providers.dstack.python",
    )
    environment_value = payload.get("environment_file")
    environment_file = (
        _configured_path(
            environment_value,
            base,
            "providers.dstack.environment_file",
        )
        if environment_value is not None
        else None
    )
    return DstackBinding(
        project,
        python,
        environment_file,
        _parse_storage(
            payload.get("storage"),
            base=base,
            context="providers.dstack.storage",
        ),
        _optional_configured_path(
            payload.get("trust_bundle"),
            base,
            "providers.dstack.trust_bundle",
        ),
        _optional_nonnegative_int(
            payload.get("capacity_wait_seconds"),
            "providers.dstack.capacity_wait_seconds",
        )
        or 0,
    )


def load_machine_config() -> MachineConfig | None:
    """Load this user's one machine-wide Posttrain configuration, if present."""

    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser().resolve()
    path = config_home / "posttrain" / "config.toml"
    if not path.exists():
        return None
    if not path.is_file():
        raise ContractError(f"Posttrain machine configuration is not a file: {path}")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"invalid Posttrain machine configuration {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ContractError("Posttrain machine configuration schema_version must be 1")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "machine_name",
            "projects",
            "default_provider",
            "defaults",
            "services",
            "huggingface",
            "credentials",
            "tracking",
            "trust",
            "storage",
            "providers",
        },
        "machine config",
    )

    machine_name = _optional_hostname(payload.get("machine_name"), "machine_name")
    if machine_name is None:
        machine_name = cast(str, _optional_hostname(socket.getfqdn(), "machine hostname"))
    projects = _absolute_path_tuple(payload.get("projects"), context="projects")
    default_provider = _optional_config_string(payload.get("default_provider"), "default_provider") or "local"
    if default_provider not in {"local", "dstack"}:
        raise ContractError("execution configuration default_provider must be 'local' or 'dstack'")
    parsed_defaults = _parse_overrides(payload.get("defaults"), context="defaults")
    if parsed_defaults.provider is not None:
        raise ContractError("execution configuration defaults.provider is replaced by top-level default_provider")
    defaults = ExecutionOverrides(
        provider=default_provider,
        target=parsed_defaults.target,
        runtime_profile=parsed_defaults.runtime_profile,
        timeout_seconds=parsed_defaults.timeout_seconds,
        max_attempts=parsed_defaults.max_attempts,
        priority=parsed_defaults.priority,
        environment_names=parsed_defaults.environment_names,
    )

    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    machine_state_root = state_home.expanduser().resolve() / "posttrain"
    storage = _parse_storage(payload.get("storage"), base=machine_state_root, context="storage")
    trust = _mapping(payload.get("trust"), context="trust", allow_none=True)
    _reject_unknown(trust, {"ca_bundle"}, "trust")
    trust_bundle = _optional_configured_path(trust.get("ca_bundle"), path.parent, "trust.ca_bundle")
    if trust_bundle is not None and not trust_bundle.is_file():
        raise ContractError(f"Posttrain machine trust bundle is missing: {trust_bundle}")
    local = LocalProviderBinding(
        canonical_hostname=machine_name,
        storage=storage,
        trust_bundle=trust_bundle,
    )

    credential_payload = _mapping(payload.get("credentials"), context="credentials", allow_none=True)
    credential_sources: dict[str, Path] = {}
    for credential_name, raw_source in credential_payload.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", credential_name):
            raise ContractError(f"invalid machine credential name: {credential_name!r}")
        source = _mapping(raw_source, context=f"credentials.{credential_name}")
        _reject_unknown(source, {"file"}, f"credentials.{credential_name}")
        source_path = _configured_path(source.get("file"), path.parent, f"credentials.{credential_name}.file")
        _require_protected_file(source_path, f"machine credential {credential_name!r}")
        credential_sources[credential_name] = source_path

    providers = _mapping(payload.get("providers"), context="providers", allow_none=True)
    _reject_unknown(providers, {"dstack"}, "providers")
    dstack_payload = _mapping(providers.get("dstack"), context="providers.dstack", allow_none=True)
    dstack: DstackBinding | None = None
    if dstack_payload:
        _reject_unknown(
            dstack_payload,
            {"project", "python", "credentials", "credentials_file", "capacity_wait_seconds"},
            "providers.dstack",
        )
        credential_name = _optional_config_string(dstack_payload.get("credentials"), "providers.dstack.credentials")
        legacy_credentials_file = dstack_payload.get("credentials_file")
        if credential_name is not None and legacy_credentials_file is not None:
            raise ContractError("providers.dstack must not set both credentials and credentials_file")
        if credential_name is not None:
            credentials_file = _credential_source(
                credential_sources,
                credential_name,
                context="providers.dstack.credentials",
            )
        elif legacy_credentials_file is not None:
            credentials_file = _configured_path(
                legacy_credentials_file,
                path.parent,
                "providers.dstack.credentials_file",
            )
            _require_protected_file(credentials_file, "dstack credentials file")
            warnings.warn(
                "providers.dstack.credentials_file is deprecated; define a named [credentials] source",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ContractError("providers.dstack.credentials must name a configured credential source")
        dstack = DstackBinding(
            project=_required_config_string(dstack_payload.get("project"), "providers.dstack.project"),
            python=_configured_executable_path(
                dstack_payload.get("python"),
                path.parent,
                "providers.dstack.python",
            ),
            environment_file=credentials_file,
            trust_bundle=trust_bundle,
            capacity_wait_seconds=_optional_nonnegative_int(
                dstack_payload.get("capacity_wait_seconds"),
                "providers.dstack.capacity_wait_seconds",
            )
            or 0,
        )
    if default_provider == "dstack" and dstack is None:
        raise ContractError("default_provider is dstack but providers.dstack is not configured")

    tracking_payload = _mapping(payload.get("tracking"), context="tracking", allow_none=True)
    _reject_unknown(tracking_payload, {"kind", "endpoint", "credentials"}, "tracking")
    tracking: MachineTrackingBinding | None = None
    if tracking_payload:
        tracking_kind = _required_config_string(tracking_payload.get("kind"), "tracking.kind")
        if tracking_kind not in {"trackio", "wandb"}:
            raise ContractError("execution configuration tracking.kind must be 'trackio' or 'wandb'")
        tracking = MachineTrackingBinding(
            kind=cast(Literal["trackio", "wandb"], tracking_kind),
            endpoint=_optional_http_url(tracking_payload.get("endpoint"), "tracking.endpoint"),
            credentials=_credential_reference(
                credential_sources,
                tracking_payload.get("credentials"),
                context="tracking.credentials",
            ),
        )

    huggingface_payload = _mapping(payload.get("huggingface"), context="huggingface", allow_none=True)
    _reject_unknown(huggingface_payload, {"credentials"}, "huggingface")
    huggingface = (
        MachineHuggingFaceBinding(
            credentials=_credential_reference(
                credential_sources,
                huggingface_payload.get("credentials"),
                context="huggingface.credentials",
            )
        )
        if huggingface_payload
        else None
    )

    services_payload = _mapping(payload.get("services"), context="services", allow_none=True)
    _reject_unknown(
        services_payload,
        {"python_index_url", "python_index_credentials", "job_registry"},
        "services",
    )
    services = MachineServicesBinding(
        python_index_url=_optional_http_url(services_payload.get("python_index_url"), "services.python_index_url"),
        python_index_credentials=_credential_reference(
            credential_sources,
            services_payload.get("python_index_credentials"),
            context="services.python_index_credentials",
        ),
        job_registry=_optional_config_string(services_payload.get("job_registry"), "services.job_registry"),
    )
    return MachineConfig(
        machine_name,
        path,
        projects,
        defaults,
        local,
        dstack,
        tracking,
        huggingface,
        services,
        MappingProxyType(credential_sources),
    )


WELL_KNOWN_TRUST_BUNDLE = Path("/etc/posttrain/trust/internal-ca.pem")
"""Where an internally-issued certificate authority is expected to be installed.

Config management puts the same file here on every machine that runs jobs, so a
project needs no trust configuration at all. It holds the internal authority
alone, never a hand-assembled union with the system store: the job image merges
it with the authorities it already has, and a union assembled here would instead
pin the job to whatever the machine that submitted it happened to trust.
"""

TRUST_BUNDLE_ENVIRONMENT_VARIABLE = "POSTTRAIN_TRUST_BUNDLE"

ADMISSION_ROOT_ENVIRONMENT_VARIABLE = "POSTTRAIN_ADMISSION_ROOT"
"""Absolute directory whose `admission/` child holds the machine-scoped ledger.

Unset, the CLI prefers `/var/lib/posttrain` when writable (worker layout), else
`$XDG_STATE_HOME/posttrain` so two projects on one laptop share host placements.
"""

_WORKER_ADMISSION_ROOT = Path("/var/lib/posttrain")


@dataclass(frozen=True, slots=True)
class ResolvedTrustBundle:
    """The additional certificate authority a job will be given, and why."""

    path: Path | None
    source: Literal["configured", "environment", "convention", "none"]


def resolve_trust_bundle(configured: Path | None) -> ResolvedTrustBundle:
    """Resolve the additional authority from configuration, environment, or convention.

    An explicitly named bundle that does not exist is an error rather than a
    reason to fall through: someone asked for a specific authority, and quietly
    substituting a different one would be worse than refusing. The convention
    path is the only source allowed to be absent, because its absence is how a
    machine says it has no internal authority.
    """
    if configured is not None:
        resolved = configured.expanduser()
        if not resolved.is_file():
            raise ContractError(f"configured providers trust_bundle does not exist: {resolved}")
        return ResolvedTrustBundle(resolved.resolve(), "configured")

    declared = os.environ.get(TRUST_BUNDLE_ENVIRONMENT_VARIABLE, "").strip()
    if declared:
        resolved = Path(declared).expanduser()
        if not resolved.is_file():
            raise ContractError(f"{TRUST_BUNDLE_ENVIRONMENT_VARIABLE} does not name a file: {resolved}")
        return ResolvedTrustBundle(resolved.resolve(), "environment")

    if WELL_KNOWN_TRUST_BUNDLE.is_file():
        return ResolvedTrustBundle(WELL_KNOWN_TRUST_BUNDLE, "convention")

    return ResolvedTrustBundle(None, "none")


def resolve_admission_state_root() -> Path:
    """Return the absolute root that owns the machine admission ledger.

    The service stores state under ``<root>/admission``. Project
    ``.posttrain/state`` still holds submission receipts; only the cross-project
    host lock lives here.
    """
    declared = os.environ.get(ADMISSION_ROOT_ENVIRONMENT_VARIABLE, "").strip()
    if declared:
        root = Path(declared).expanduser()
        if not root.is_absolute():
            raise ContractError(f"{ADMISSION_ROOT_ENVIRONMENT_VARIABLE} must be an absolute path: {root}")
        return root.resolve()

    if _WORKER_ADMISSION_ROOT.is_dir() and os.access(_WORKER_ADMISSION_ROOT, os.W_OK):
        return _WORKER_ADMISSION_ROOT.resolve()

    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser() / "posttrain").resolve()
    return (Path.home() / ".local" / "state" / "posttrain").resolve()


def configured_registry_prefix(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the project's registry prefix from the environment, if set."""
    values = os.environ if environ is None else environ
    raw = values.get(REGISTRY_ENVIRONMENT_VARIABLE, "").strip().rstrip("/")
    return raw or None


def _published_manifest() -> PublishedManifest:
    try:
        return load_manifest()
    except ManifestError as error:
        raise ContractError(f"installed runtime image manifest is unusable: {error}") from error


def _derived_kind_images(
    manifest: PublishedManifest,
    *,
    mirror_prefix: str | None,
) -> dict[str, RuntimeImageRef]:
    """Resolve every published variant to a digest-pinned reference.

    A mirror preserves image digests, so mirroring only changes the prefix.
    Without one, images resolve to the framework's own release registry.
    """
    prefix = mirror_prefix or manifest.default_prefix
    return {variant: RuntimeImageRef(image.reference(prefix)) for variant, image in manifest.kinds.items()}


def _derived_constraint_profiles(
    manifest: PublishedManifest,
) -> dict[str, ConstraintProfileBinding]:
    """Bind each published variant to the lock its image was built from.

    The digest is not read back from the manifest: `load_manifest` has already
    checked each recorded lock digest against the shipped bytes, so a lock
    edited without republishing cannot reach this point.
    """
    root = cached_definition_root()
    profiles: dict[str, ConstraintProfileBinding] = {}
    for variant, image in manifest.kinds.items():
        path = root / image.constraint_lock
        constraints = KindDependencyConstraints(
            variant,
            path.read_text(encoding="utf-8"),
            image.provided_packages,
        )
        backend_path: Path | None = None
        backend_contents_digest: str | None = None
        backend_provided_packages: tuple[str, ...] = ()
        backend_digest: str | None = None
        if image.backend_constraint_lock is not None:
            # A second locked environment inside the same image, published by
            # the release exactly like the control lock. Deriving it here is
            # what lets a project with no machine binding pack a veRL job.
            backend_path = root / image.backend_constraint_lock
            backend_contents_digest = image.backend_lock_digest
            backend = KindDependencyConstraints(
                variant,
                backend_path.read_text(encoding="utf-8"),
                image.backend_provided_packages,
            )
            backend_provided_packages = backend.provided_packages
            backend_digest = KindDependencyConstraints(
                variant,
                backend.contents,
                backend_provided_packages,
                role="backend",
                python_version="3.13.12",
                python_executable="/opt/posttrain-verl/bin/python",
                requirements_filename="runtime.backend.requirements.txt",
            ).digest
        profiles[variant] = ConstraintProfileBinding(
            path,
            image.lock_digest,
            constraints.provided_packages,
            constraints.digest,
            backend_path,
            backend_contents_digest,
            backend_provided_packages,
            backend_digest,
        )
    return profiles


def _resolved_repository(
    declared: object,
    *,
    context: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    if declared is not None:
        return _required_config_string(declared, context)
    prefix = configured_registry_prefix(environ)
    if prefix is None:
        raise ContractError(
            "job packing needs a registry for this project's actual-job images: "
            f"set {REGISTRY_ENVIRONMENT_VARIABLE} to an OCI registry prefix you can "
            f"push to, or declare [registry].repository in the execution configuration"
        )
    return f"{prefix}/{_JOB_REPOSITORY_SUFFIX}"


def derived_registry(environ: Mapping[str, str] | None = None) -> RegistryBinding | None:
    """Build a registry binding with no execution configuration file at all."""
    if configured_registry_prefix(environ) is None:
        return None
    return _parse_registry({}, base=Path.cwd(), environ=environ)


def derived_local_registry() -> RegistryBinding:
    """Resolve shipped image and lock identities for a non-publishing OCI export."""

    registry = _parse_registry(
        {"repository": "posttrain.local/posttrain/jobs"},
        base=Path.cwd(),
        environ={},
    )
    assert registry is not None
    return registry


def _parse_registry(
    value: object,
    *,
    base: Path,
    environ: Mapping[str, str] | None = None,
) -> RegistryBinding | None:
    if value is None:
        return None
    payload = _mapping(value, context="registry")
    _reject_unknown(
        payload,
        {
            "repository",
            "universal_image",
            "kind_images",
            "constraint_profiles",
            "mirror_prefix",
            "buildx_builder",
            "receipt_root",
            "bake_file",
            "framework_source_root",
        },
        "registry",
    )
    receipt_value = payload.get("receipt_root")
    bake_value = payload.get("bake_file")
    framework_source_value = payload.get("framework_source_root")
    mirror_prefix = _optional_config_string(
        payload.get("mirror_prefix"),
        "registry.mirror_prefix",
    )
    manifest = _published_manifest()

    # Explicit declarations win per variant; everything else comes from the
    # installed manifest. This is what removes the hand transcription: a
    # consumer no longer restates digests or lock hashes the release already
    # fixed.
    kind_images = dict(_derived_kind_images(manifest, mirror_prefix=mirror_prefix))
    if payload.get("kind_images") is not None:
        kind_images.update(
            _runtime_image_mapping(
                payload.get("kind_images"),
                context="registry.kind_images",
            )
        )
    constraint_profiles = dict(_derived_constraint_profiles(manifest))
    if payload.get("constraint_profiles") is not None:
        constraint_profiles.update(
            _constraint_profile_mapping(
                payload.get("constraint_profiles"),
                base=base,
            )
        )
    if set(kind_images) != set(constraint_profiles):
        raise ContractError(
            "execution configuration registry kind_images and constraint_profiles must define the same runtime variants"
        )
    universal_value = payload.get("universal_image")
    return RegistryBinding(
        repository=_resolved_repository(
            payload.get("repository"),
            context="registry.repository",
            environ=environ,
        ),
        universal_image=RuntimeImageRef(
            _required_config_string(universal_value, "registry.universal_image")
            if universal_value is not None
            else manifest.base.reference(mirror_prefix or manifest.default_prefix)
        ),
        kind_images=MappingProxyType(kind_images),
        constraint_profiles=MappingProxyType(constraint_profiles),
        mirror_prefix=mirror_prefix,
        buildx_builder=_optional_config_string(
            payload.get("buildx_builder"),
            "registry.buildx_builder",
        ),
        receipt_root=(
            _configured_path(receipt_value, base, "registry.receipt_root") if receipt_value is not None else None
        ),
        bake_file=(_configured_path(bake_value, base, "registry.bake_file") if bake_value is not None else None),
        framework_source_root=(
            _configured_path(
                framework_source_value,
                base,
                "registry.framework_source_root",
            )
            if framework_source_value is not None
            else None
        ),
    )


def _runtime_image_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, RuntimeImageRef]:
    payload = _mapping(value, context=context)
    _require_profile_keys(payload, context)
    return MappingProxyType(
        {
            profile: RuntimeImageRef(_required_config_string(payload[profile], f"{context}.{profile}"))
            for profile in sorted(payload)
        }
    )


def _constraint_profile_mapping(
    value: object,
    *,
    base: Path,
) -> Mapping[str, ConstraintProfileBinding]:
    payload = _mapping(value, context="registry.constraint_profiles")
    _require_profile_keys(payload, "registry.constraint_profiles")
    parsed: dict[str, ConstraintProfileBinding] = {}
    for profile in sorted(payload):
        context = f"registry.constraint_profiles.{profile}"
        item = _mapping(payload[profile], context=context)
        _reject_unknown(
            item,
            {
                "path",
                "sha256",
                "provided_packages",
                "backend_path",
                "backend_sha256",
            },
            context,
        )
        path = _configured_path(item.get("path"), base, f"{context}.path")
        if not path.is_file():
            raise ContractError(f"execution configuration {context}.path is missing: {path}")
        expected = _required_config_string(item.get("sha256"), f"{context}.sha256")
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ContractError(f"execution configuration {context}.sha256 must be lowercase SHA-256")
        observed = sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ContractError(f"execution configuration {context} differs from its exact digest")
        provided_packages = (
            _string_tuple(
                item.get("provided_packages"),
                context=f"{context}.provided_packages",
                allow_none=True,
            )
            or ()
        )
        constraints = KindDependencyConstraints(
            profile,
            path.read_text(encoding="utf-8"),
            provided_packages,
        )
        backend_path_value = item.get("backend_path")
        backend_digest_value = item.get("backend_sha256")
        if (backend_path_value is None) != (backend_digest_value is None):
            raise ContractError(
                f"execution configuration {context} must declare backend_path and backend_sha256 together"
            )
        backend_path: Path | None = None
        backend_contents_digest: str | None = None
        backend_provided_packages: tuple[str, ...] = ()
        backend_digest: str | None = None
        if backend_path_value is not None:
            backend_path = _configured_path(
                backend_path_value,
                base,
                f"{context}.backend_path",
            )
            if not backend_path.is_file():
                raise ContractError(f"execution configuration {context}.backend_path is missing: {backend_path}")
            backend_contents_digest = _required_config_string(
                backend_digest_value,
                f"{context}.backend_sha256",
            )
            if len(backend_contents_digest) != 64 or any(
                character not in "0123456789abcdef" for character in backend_contents_digest
            ):
                raise ContractError(f"execution configuration {context}.backend_sha256 must be lowercase SHA-256")
            observed_backend = sha256(backend_path.read_bytes()).hexdigest()
            if observed_backend != backend_contents_digest:
                raise ContractError(f"execution configuration {context}.backend_path differs from its exact digest")
            backend_constraints = KindDependencyConstraints(
                profile,
                backend_path.read_text(encoding="utf-8"),
            )
            backend_provided_packages = backend_constraints.constrained_packages
            backend_constraints = KindDependencyConstraints(
                profile,
                backend_constraints.contents,
                backend_provided_packages,
                role="backend",
                python_version="3.13.12",
                python_executable="/opt/posttrain-verl/bin/python",
                requirements_filename="runtime.backend.requirements.txt",
            )
            backend_digest = backend_constraints.digest
        parsed[profile] = ConstraintProfileBinding(
            path,
            expected,
            constraints.provided_packages,
            constraints.digest,
            backend_path,
            backend_contents_digest,
            backend_provided_packages,
            backend_digest,
        )
    return MappingProxyType(parsed)


def _require_profile_keys(payload: Mapping[str, object], context: str) -> None:
    observed = set(payload)
    if not observed:
        raise ContractError(f"execution configuration {context} must define at least one runtime variant")
    invalid = sorted(value for value in observed if re.fullmatch(r"[a-z0-9][a-z0-9-]*", value) is None)
    if invalid:
        raise ContractError(f"execution configuration {context} has invalid runtime variants: " + ", ".join(invalid))


def _parse_storage(
    value: object,
    *,
    base: Path,
    context: str,
) -> ExecutionStorageBinding | None:
    if value is None:
        return None
    payload = _mapping(value, context=context)
    _reject_unknown(payload, {"run_root", "model_cache", "compile_cache"}, context)
    run_root = _configured_path(payload.get("run_root"), base, f"{context}.run_root")
    model_value = payload.get("model_cache")
    compile_value = payload.get("compile_cache")
    return ExecutionStorageBinding(
        run_root=run_root,
        model_cache=(
            _configured_path(model_value, base, f"{context}.model_cache") if model_value is not None else None
        ),
        compile_cache=(
            _configured_path(compile_value, base, f"{context}.compile_cache") if compile_value is not None else None
        ),
    )


def _mapping(
    value: object,
    *,
    context: str,
    allow_none: bool = False,
) -> dict[str, object]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"execution configuration {context} must be a table")
    return value


def _reject_unknown(
    payload: dict[str, object],
    allowed: set[str],
    context: str,
) -> None:
    if unknown := sorted(set(payload) - allowed):
        raise ContractError(f"execution configuration {context} has unknown fields: {', '.join(unknown)}")


def _required_config_string(value: object, context: str) -> str:
    parsed = _optional_config_string(value, context)
    if parsed is None:
        raise ContractError(f"execution configuration {context} is required")
    return parsed


def _optional_config_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"execution configuration {context} must be a non-empty string")
    return value


def _credential_source(
    sources: Mapping[str, Path],
    name: str,
    *,
    context: str,
) -> Path:
    try:
        return sources[name]
    except KeyError as error:
        raise ContractError(
            f"execution configuration {context} references unknown credential source {name!r}"
        ) from error


def _credential_reference(
    sources: Mapping[str, Path],
    value: object,
    *,
    context: str,
) -> str | None:
    name = _optional_config_string(value, context)
    if name is not None:
        _credential_source(sources, name, context=context)
    return name


def _optional_http_url(value: object, context: str) -> str | None:
    parsed = _optional_config_string(value, context)
    if parsed is None:
        return None
    url = urlparse(parsed)
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ContractError(f"execution configuration {context} must be a credential-free HTTP(S) URL")
    return parsed


def _optional_hostname(value: object, context: str) -> str | None:
    parsed = _optional_config_string(value, context)
    if parsed is None:
        return None
    hostname = parsed.strip().lower().rstrip(".")
    if len(hostname) > 253 or any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or re.fullmatch(r"[a-z0-9-]+", label) is None
        for label in hostname.split(".")
    ):
        raise ContractError(f"execution configuration {context} must be a canonical hostname")
    return hostname


def _configured_path(value: object, base: Path, context: str) -> Path:
    configured = Path(_required_config_string(value, context)).expanduser()
    return (configured if configured.is_absolute() else base / configured).resolve()


def _optional_configured_path(
    value: object,
    base: Path,
    context: str,
) -> Path | None:
    return _configured_path(value, base, context) if value is not None else None


def _configured_executable_path(value: object, base: Path, context: str) -> Path:
    """Make an executable absolute without collapsing virtualenv symlinks."""

    configured = Path(_required_config_string(value, context)).expanduser()
    candidate = configured if configured.is_absolute() else base / configured
    return candidate.absolute()


def _optional_positive_int(value: object, context: str) -> int | None:
    parsed = _optional_int(value, context)
    if parsed is not None and parsed < 1:
        raise ContractError(f"execution configuration {context} must be positive")
    return parsed


def _optional_nonnegative_int(value: object, context: str) -> int | None:
    parsed = _optional_int(value, context)
    if parsed is not None and parsed < 0:
        raise ContractError(f"execution configuration {context} must not be negative")
    return parsed


def _optional_int(value: object, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"execution configuration {context} must be an integer")
    return value


def _string_tuple(
    value: object,
    *,
    context: str,
    allow_none: bool = False,
) -> tuple[str, ...] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"execution configuration {context} must be a string array")
    parsed = tuple(value)
    if len(set(parsed)) != len(parsed) or any(not item.strip() or "=" in item for item in parsed):
        raise ContractError(f"execution configuration {context} must contain unique variable names")
    return parsed


def _absolute_path_tuple(value: object, *, context: str) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"execution configuration {context} must be an absolute path array")
    paths = tuple(Path(item).expanduser() for item in value)
    if any(not path.is_absolute() for path in paths):
        raise ContractError(f"execution configuration {context} must contain only absolute paths")
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ContractError(f"execution configuration {context} must contain unique paths")
    return resolved


def load_execution_environment(
    configuration: LocalExecutionConfig,
) -> dict[str, str]:
    """Overlay project runtime values on reusable machine service defaults."""

    environment: dict[str, str] = {}
    if configuration.machine is not None:
        machine = configuration.machine
        if machine.tracking is not None and machine.tracking.endpoint is not None:
            name = "POSTTRAIN_TRACKIO_SERVER_URL" if machine.tracking.kind == "trackio" else "WANDB_BASE_URL"
            environment[name] = machine.tracking.endpoint
        if machine.tracking is not None and machine.tracking.credentials is not None:
            _merge_credential_environment(
                environment,
                machine.credentials[machine.tracking.credentials],
                allowed=({"TRACKIO_WRITE_TOKEN"} if machine.tracking.kind == "trackio" else {"WANDB_API_KEY"}),
                purpose="tracking",
            )
        if machine.huggingface is not None and machine.huggingface.credentials is not None:
            _merge_credential_environment(
                environment,
                machine.credentials[machine.huggingface.credentials],
                allowed={"HF_TOKEN"},
                purpose="Hugging Face",
            )
        if machine.services.python_index_url is not None:
            environment["UV_INDEX_URL"] = machine.services.python_index_url
        if machine.services.python_index_credentials is not None:
            _merge_credential_environment(
                environment,
                machine.credentials[machine.services.python_index_credentials],
                allowed={"PIP_INDEX_URL", "UV_INDEX_PASSWORD", "UV_INDEX_USERNAME"},
                purpose="Python index",
            )
        if machine.services.job_registry is not None:
            environment[REGISTRY_ENVIRONMENT_VARIABLE] = machine.services.job_registry
    path = configuration.environment_file
    if path is None:
        return environment
    environment.update(_read_environment_file(path, label="execution environment file"))
    if "POSTTRAIN_PROFILE" in environment:
        raise ContractError(
            "POSTTRAIN_PROFILE was removed; machine defaults load automatically "
            "from $XDG_CONFIG_HOME/posttrain/config.toml"
        )
    return environment


def _read_environment_file(path: Path, *, label: str) -> dict[str, str]:
    _require_protected_file(path, label)
    environment: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.partition("=")
        if not separator or not name.strip():
            raise ContractError(f"invalid execution environment file: {path}")
        parsed = shlex.split(raw_value, comments=False, posix=True)
        if len(parsed) != 1:
            raise ContractError(f"invalid execution environment value for {name.strip()}")
        environment[name.strip()] = parsed[0]
    return environment


def _merge_credential_environment(
    environment: dict[str, str],
    path: Path,
    *,
    allowed: set[str],
    purpose: str,
) -> None:
    values = _read_environment_file(path, label=f"{purpose} credential file")
    if unknown := sorted(set(values) - allowed):
        raise ContractError(
            f"{purpose} credential file {path} contains variables outside its scope: " + ", ".join(unknown)
        )
    for name, value in values.items():
        if name in environment and environment[name] != value:
            raise ContractError(f"machine credential sources disagree on {name}")
        environment[name] = value


def _require_protected_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ContractError(f"{label} is missing: {path}")
    if path.stat().st_mode & 0o077:
        raise ContractError(f"{label} must not be accessible by group or others: {path}")


__all__ = [
    "ConstraintProfileBinding",
    "DstackBinding",
    "ExecutionOverrides",
    "ExecutionStorageBinding",
    "LaunchOverrides",
    "LocalProviderBinding",
    "LocalExecutionConfig",
    "MachineHuggingFaceBinding",
    "PackageOverrides",
    "RegistryBinding",
    "ResolvedExecutionSettings",
    "SettingSource",
    "MachineConfig",
    "MachineServicesBinding",
    "MachineTrackingBinding",
    "load_local_execution_config",
    "load_machine_config",
    "load_execution_environment",
    "resolve_execution_settings",
]
