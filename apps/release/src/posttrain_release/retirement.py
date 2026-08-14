"""Audited retirement of a failed, never-promoted release candidate."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

from .artifacts import _authenticated_request, verify_index_receipt


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.names: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = next((value for name, value in attrs if name == "href"), None)
        if href:
            self.names.add(unquote(urlparse(href).path.rsplit("/", 1)[-1]))


def _load_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "posttrain.python-release-receipt.v1":
        raise ValueError(f"unsupported Python release receipt: {path}")
    for key in ("version", "revision"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise ValueError(f"release receipt has no {key}: {path}")
    packages = receipt.get("packages")
    artifacts = receipt.get("artifacts")
    if not isinstance(packages, list) or not packages or not all(isinstance(item, str) for item in packages):
        raise ValueError(f"release receipt packages are invalid: {path}")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"release receipt artifacts are invalid: {path}")
    return receipt


def _receipt_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_names(receipt: dict[str, object]) -> dict[str, list[str]]:
    version = str(receipt["version"])
    artifacts = receipt["artifacts"]
    packages = receipt["packages"]
    assert isinstance(artifacts, list)
    assert isinstance(packages, list)
    by_package: dict[str, list[str]] = {}
    for package in packages:
        assert isinstance(package, str)
        distribution = re.sub(r"[-.]+", "_", package)
        wheel_prefix = f"{distribution}-{version}-"
        sdist_name = f"{distribution}-{version}.tar.gz"
        names = sorted(
            str(item["filename"])
            for item in artifacts
            if isinstance(item, dict)
            and isinstance(item.get("filename"), str)
            and (str(item["filename"]).startswith(wheel_prefix) or item["filename"] == sdist_name)
        )
        if len(names) != 2:
            raise ValueError(f"release receipt must contain one wheel and sdist for {package} {version}")
        by_package[package] = names
    if sum(len(names) for names in by_package.values()) != len(artifacts):
        raise ValueError("release receipt contains artifacts outside its coordinated package set")
    return by_package


def index_version_artifacts(receipt: dict[str, object], simple_base_url: str) -> dict[str, list[str]]:
    """Return every file for this exact coordinated version on each package page."""

    version = str(receipt["version"])
    packages = receipt["packages"]
    assert isinstance(packages, list)
    base = simple_base_url.rstrip("/") + "/"
    observed: dict[str, list[str]] = {}
    for package in packages:
        assert isinstance(package, str)
        normalized = re.sub(r"[-_.]+", "-", package).lower()
        distribution = re.sub(r"[-.]+", "_", package)
        page_url = urljoin(base, quote(normalized) + "/")
        try:
            with urllib.request.urlopen(_authenticated_request(page_url), timeout=20) as response:  # noqa: S310
                document = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                observed[package] = []
                continue
            raise RuntimeError(f"cannot read package index page for {package}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"cannot read package index page for {package}") from error
        parser = _Links()
        parser.feed(document)
        wheel_prefix = f"{distribution}-{version}-"
        sdist_name = f"{distribution}-{version}.tar.gz"
        observed[package] = sorted(
            name for name in parser.names if name.startswith(wheel_prefix) or name == sdist_name
        )
    return observed


def create_retirement_preflight(
    failed_receipt_path: Path,
    replacement_receipt_path: Path,
    *,
    failed_run_id: str,
    development_simple_url: str,
    stable_simple_url: str,
) -> dict[str, object]:
    """Prove that one failed development version is safe to remove."""

    failed = _load_receipt(failed_receipt_path)
    replacement = _load_receipt(replacement_receipt_path)
    if failed["version"] != replacement["version"]:
        raise ValueError("failed and replacement receipts have different versions")
    if failed["revision"] == replacement["revision"]:
        raise ValueError("replacement candidate must come from a different source revision")
    if failed["packages"] != replacement["packages"]:
        raise ValueError("failed and replacement receipts have different coordinated package sets")

    expected = _artifact_names(failed)
    verify_index_receipt(failed_receipt_path, development_simple_url)
    development = index_version_artifacts(failed, development_simple_url)
    if development != expected:
        raise ValueError("development version is partial, contains extra files, or differs from the failed receipt")
    stable = index_version_artifacts(failed, stable_simple_url)
    present_in_stable = {package: names for package, names in stable.items() if names}
    if present_in_stable:
        raise ValueError(f"failed candidate version already exists in stable: {sorted(present_in_stable)}")

    return {
        "schema": "posttrain.failed-candidate-retirement-preflight.v1",
        "status": "verified-for-deletion",
        "failed_run_id": failed_run_id,
        "version": failed["version"],
        "failed_revision": failed["revision"],
        "replacement_revision": replacement["revision"],
        "failed_receipt_sha256": _receipt_digest(failed_receipt_path),
        "replacement_receipt_sha256": _receipt_digest(replacement_receipt_path),
        "packages": failed["packages"],
        "artifacts": failed["artifacts"],
    }


def create_retirement_completion(
    failed_receipt_path: Path,
    replacement_receipt_path: Path,
    preflight_path: Path,
    *,
    development_simple_url: str,
    stable_simple_url: str,
) -> dict[str, object]:
    """Prove deletion completed without publishing anything to stable."""

    failed = _load_receipt(failed_receipt_path)
    replacement = _load_receipt(replacement_receipt_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("schema") != "posttrain.failed-candidate-retirement-preflight.v1":
        raise ValueError("unsupported failed-candidate retirement preflight")
    checks = {
        "version": failed["version"],
        "failed_revision": failed["revision"],
        "replacement_revision": replacement["revision"],
        "failed_receipt_sha256": _receipt_digest(failed_receipt_path),
        "replacement_receipt_sha256": _receipt_digest(replacement_receipt_path),
        "packages": failed["packages"],
    }
    for key, expected in checks.items():
        if preflight.get(key) != expected:
            raise ValueError(f"retirement preflight {key} does not match current receipts")
    development = index_version_artifacts(failed, development_simple_url)
    stable = index_version_artifacts(failed, stable_simple_url)
    if any(development.values()):
        raise ValueError("failed candidate version remains in development after deletion")
    if any(stable.values()):
        raise ValueError("failed candidate version appeared in stable during retirement")
    return {
        "schema": "posttrain.failed-candidate-retirement.v1",
        "status": "retired",
        **checks,
        "failed_run_id": preflight["failed_run_id"],
        "preflight_sha256": _receipt_digest(preflight_path),
    }


def write_retirement_receipt(receipt: dict[str, object], destination: Path) -> None:
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "create_retirement_completion",
    "create_retirement_preflight",
    "index_version_artifacts",
    "write_retirement_receipt",
]
