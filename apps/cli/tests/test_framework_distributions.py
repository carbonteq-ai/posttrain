"""What gets staged into an actual-job image when there is no checkout."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

from posttrain_cli.framework_distributions import (
    FRAMEWORK_DISTRIBUTIONS,
    IMAGE_ONLY_DISTRIBUTION,
    _index_arguments,
    installed_versions,
    materialize,
)


def test_the_image_entry_point_is_staged_even_though_nobody_installs_it() -> None:
    """`posttrain-runtime` runs inside the image and nowhere else.

    A consumer installs `posttrain` and never has a reason to install the
    runtime, so reading the staged set from the environment alone produces an
    image whose ENTRYPOINT does not exist. The failure appears only when the
    image is built, as `posttrain-runtime: not found`.
    """
    versions = installed_versions()
    assert IMAGE_ONLY_DISTRIBUTION in versions
    assert versions[IMAGE_ONLY_DISTRIBUTION] == versions["posttrain"]
    assert IMAGE_ONLY_DISTRIBUTION in FRAMEWORK_DISTRIBUTIONS
    assert "posttrain-environment" in FRAMEWORK_DISTRIBUTIONS
    assert "posttrain-project" in FRAMEWORK_DISTRIBUTIONS


def test_staged_distributions_include_their_framework_dependencies() -> None:
    """A job image must carry every framework package its staged packages need.

    The list is explicit, so a new framework package (``posttrain-advisor``,
    imported by ``posttrain-work``) is missing from job images until it is
    added, and the image fails at build time with ``ModuleNotFoundError``.
    """
    # Needed on the host or behind an optional selection, never inside a job:
    # the job-image builder runs where the image is built, and W&B tracking is
    # imported only when selected (job-kind locks carry no ``wandb``).
    not_in_jobs = {"posttrain-execution-job-builder", "posttrain-tracking-wandb"}
    missing: dict[str, list[str]] = {}
    for distribution in FRAMEWORK_DISTRIBUTIONS:
        for requirement in importlib.metadata.requires(distribution) or ():
            name, _, marker = requirement.partition(";")
            if "extra ==" in marker:
                continue
            dependency = re.split(r"[\s\[<>=!~@(]", name.strip(), maxsplit=1)[0].lower().replace("_", "-")
            if dependency.startswith("posttrain") and dependency not in {*FRAMEWORK_DISTRIBUTIONS, *not_in_jobs}:
                missing.setdefault(distribution, []).append(dependency)
    assert missing == {}


def test_explicit_wheelhouse_is_copied_and_digest_bound(tmp_path: Path, monkeypatch) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "posttrain-0.3.0-py3-none-any.whl").write_bytes(b"cli")
    (wheelhouse / "posttrain_runtime-0.3.0-py3-none-any.whl").write_bytes(b"runtime")
    monkeypatch.setattr(
        "posttrain_cli.framework_distributions.installed_versions",
        lambda: {"posttrain": "0.3.0", "posttrain-runtime": "0.3.0"},
    )

    resolved = materialize(tmp_path / "staged", wheelhouse=wheelhouse)

    assert resolved.filenames == (
        "posttrain-0.3.0-py3-none-any.whl",
        "posttrain_runtime-0.3.0-py3-none-any.whl",
    )
    assert resolved.digest


def test_a_uv_only_consumer_still_has_their_index_used() -> None:
    """pip reads PIP_INDEX_URL; the documented setup only sets the uv variable."""
    index = "https://pypi.example.invalid/simple/"
    assert _index_arguments({"UV_INDEX_URL": index}) == ["--index-url", index]
    assert _index_arguments({"UV_DEFAULT_INDEX": index}) == ["--index-url", index]
    assert _index_arguments({}) == []


def test_an_explicit_pip_index_is_left_alone() -> None:
    """A consumer who configured pip directly has already said what they want."""
    assert _index_arguments({"PIP_INDEX_URL": "https://pip.example.invalid/simple/"}) == []
