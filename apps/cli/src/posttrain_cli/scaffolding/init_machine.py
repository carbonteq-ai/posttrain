"""Initialize reusable configuration for one developer or controller machine."""

from __future__ import annotations

import json
import os
import ssl
import tomllib
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError


@dataclass(frozen=True, slots=True)
class MachineInitialization:
    config: Path
    credential_files: tuple[Path, ...]


def machine_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser().resolve()
    return config_home / "posttrain" / "config.toml"


def initialize_machine(
    *,
    projects: tuple[Path, ...] = (),
    machine_name: str | None = None,
    default_provider: str = "local",
    trackio_endpoint: str | None = None,
    python_index_url: str | None = None,
    job_registry: str | None = None,
    dstack_project: str | None = None,
    dstack_python: Path | None = None,
) -> MachineInitialization:
    """Write one non-secret config and narrowly scoped protected sources."""

    if default_provider not in {"local", "dstack"}:
        raise ContractError("machine default provider must be local or dstack")
    if default_provider == "dstack" and dstack_project is None:
        raise ContractError("--default-provider dstack requires --dstack-project")
    if dstack_project is not None and dstack_python is None:
        raise ContractError("--dstack-project requires --dstack-python")

    resolved_projects = tuple(path.expanduser().resolve() for path in projects)
    if len(set(resolved_projects)) != len(resolved_projects):
        raise ContractError("machine projects must be unique")
    for project in resolved_projects:
        if not (project / ".posttrain" / "project.toml").is_file():
            raise ContractError(f"machine project is not initialized: {project}")

    config = machine_config_path()
    root = config.parent
    if config.exists():
        raise FileExistsError(f"refusing to overwrite existing machine configuration: {config}")

    credential_root = root / "credentials"
    credential_root.mkdir(parents=True, exist_ok=True)
    credential_specs = {
        "trackio.env": "# TRACKIO_WRITE_TOKEN=\n",
        "huggingface.env": "# HF_TOKEN=\n",
        "python-index.env": (
            "# Optional credential-bearing index URL used only during package resolution.\n"
            "# PIP_INDEX_URL=https://USER:TOKEN@pypi.example/simple/\n"
        ),
        "dstack.env": "# DSTACK_TOKEN=\n",
    }
    credential_files: list[Path] = []
    for filename, contents in credential_specs.items():
        destination = credential_root / filename
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing machine credential file: {destination}")
        destination.write_text(contents, encoding="utf-8")
        destination.chmod(0o600)
        credential_files.append(destination)

    lines = [
        "schema_version = 1",
        f"default_provider = {json.dumps(default_provider)}",
    ]
    if machine_name is not None:
        resolved_name = machine_name.strip().lower().rstrip(".")
        lines.insert(1, f"machine_name = {json.dumps(resolved_name)}")
    if resolved_projects:
        lines.append("projects = [" + ", ".join(json.dumps(str(path)) for path in resolved_projects) + "]")
    lines.extend(
        (
            "",
            "[storage]",
            'run_root = "runs"',
            'model_cache = "cache/huggingface"',
            'compile_cache = "cache/compile"',
        )
    )

    ca_file = ssl.get_default_verify_paths().cafile
    if ca_file and Path(ca_file).is_file():
        lines.extend(("", "[trust]", f"ca_bundle = {json.dumps(ca_file)}"))
    if trackio_endpoint is not None:
        lines.extend(
            (
                "",
                "[tracking]",
                'kind = "trackio"',
                f"endpoint = {json.dumps(trackio_endpoint)}",
                'credentials = "trackio-default"',
            )
        )
    if python_index_url is not None or job_registry is not None:
        lines.extend(("", "[services]"))
        if python_index_url is not None:
            lines.extend(
                (
                    f"python_index_url = {json.dumps(python_index_url)}",
                    'python_index_credentials = "python-index-default"',
                )
            )
        if job_registry is not None:
            lines.append(f"job_registry = {json.dumps(job_registry)}")
    lines.extend(("", "[huggingface]", 'credentials = "huggingface-default"'))
    lines.extend(
        (
            "",
            "[providers.local]",
            "# Optional literal DNS servers for containers on this machine.",
            '# dns_servers = ["192.0.2.53"]',
        )
    )
    if dstack_project is not None:
        assert dstack_python is not None
        lines.extend(
            (
                "",
                "[providers.dstack]",
                f"project = {json.dumps(dstack_project)}",
                f"python = {json.dumps(str(dstack_python.expanduser().absolute()))}",
                'credentials = "dstack-default"',
            )
        )
    lines.extend(
        (
            "",
            "[credentials.trackio-default]",
            'file = "credentials/trackio.env"',
            "",
            "[credentials.huggingface-default]",
            'file = "credentials/huggingface.env"',
            "",
            "[credentials.python-index-default]",
            'file = "credentials/python-index.env"',
            "",
            "[credentials.dstack-default]",
            'file = "credentials/dstack.env"',
            "",
        )
    )
    config.write_text("\n".join(lines), encoding="utf-8")
    config.chmod(0o644)
    return MachineInitialization(config=config, credential_files=tuple(credential_files))


def add_machine_project(project: Path) -> tuple[Path, bool]:
    """Idempotently register one project without rewriting unrelated TOML."""

    resolved = project.expanduser().resolve()
    if not (resolved / ".posttrain" / "project.toml").is_file():
        raise ContractError(f"machine project is not initialized: {resolved}")
    config = machine_config_path()
    if not config.is_file():
        raise ContractError("machine configuration is missing; run posttrain machine init first")
    source = config.read_text(encoding="utf-8")
    try:
        payload = tomllib.loads(source)
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"invalid Posttrain machine configuration {config}: {error}") from error
    raw_projects = payload.get("projects", [])
    if not isinstance(raw_projects, list) or not all(isinstance(item, str) for item in raw_projects):
        raise ContractError("machine configuration projects must be an absolute path array")
    projects = [str(Path(item).expanduser().resolve()) for item in raw_projects]
    if str(resolved) in projects:
        return config, False
    projects.append(str(resolved))
    assignment = "projects = [" + ", ".join(json.dumps(item) for item in projects) + "]"

    lines = source.splitlines()
    project_lines = [index for index, line in enumerate(lines) if line.lstrip().startswith("projects =")]
    if len(project_lines) > 1:
        raise ContractError("machine configuration has more than one top-level projects assignment")
    if project_lines:
        index = project_lines[0]
        try:
            tomllib.loads(lines[index])
        except tomllib.TOMLDecodeError as error:
            raise ContractError(
                "posttrain machine project add currently requires projects to use one TOML line"
            ) from error
        lines[index] = assignment
    else:
        first_table = next((index for index, line in enumerate(lines) if line.lstrip().startswith("[")), len(lines))
        insertion = first_table
        while insertion > 0 and not lines[insertion - 1].strip():
            insertion -= 1
        lines.insert(insertion, assignment)
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config, True


__all__ = ["MachineInitialization", "add_machine_project", "initialize_machine", "machine_config_path"]
