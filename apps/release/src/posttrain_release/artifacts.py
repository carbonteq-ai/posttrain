"""Hash-addressed Python distribution receipts and private-index readback."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse


@dataclass(frozen=True, slots=True)
class DistributionArtifact:
    filename: str
    sha256: str
    size: int


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = next((value for name, value in attrs if name == "href"), None)
        if href:
            self.links[unquote(urlparse(href).path.rsplit("/", 1)[-1])] = href


def _wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError(f"wheel must contain exactly one METADATA file: {path.name}")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"wheel metadata is missing Name or Version: {path.name}")
    return name, version


def create_distribution_receipt(
    distribution_root: Path,
    *,
    version: str,
    revision: str,
    uv_lock: Path,
    image_manifest: Path,
) -> dict[str, object]:
    files = tuple(sorted(path for path in distribution_root.iterdir() if path.is_file()))
    wheels = tuple(path for path in files if path.suffix == ".whl")
    if not wheels:
        raise ValueError("release receipt requires at least one wheel")
    packages: list[str] = []
    for wheel in wheels:
        name, observed_version = _wheel_identity(wheel)
        if observed_version != version:
            raise ValueError(f"{wheel.name}: version is {observed_version!r}, expected {version!r}")
        packages.append(name)
    if len(packages) != len(set(packages)):
        raise ValueError("release receipt contains duplicate wheel package names")
    artifacts = tuple(
        DistributionArtifact(
            filename=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size=path.stat().st_size,
        )
        for path in files
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    )
    if len(artifacts) != len(wheels) * 2:
        raise ValueError("release receipt requires one wheel and one source distribution per package")
    return {
        "schema": "posttrain.python-release-receipt.v1",
        "version": version,
        "revision": revision,
        "packages": sorted(packages),
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "uv_lock_sha256": hashlib.sha256(uv_lock.read_bytes()).hexdigest(),
        "image_manifest_sha256": hashlib.sha256(image_manifest.read_bytes()).hexdigest(),
    }


def write_distribution_receipt(receipt: dict[str, object], destination: Path) -> None:
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_distribution_receipt(receipt_path: Path, distribution_root: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "posttrain.python-release-receipt.v1":
        raise ValueError("unsupported Python release receipt schema")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release receipt has no artifacts")
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ValueError("release receipt artifact is invalid")
        path = distribution_root / item["filename"]
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != item.get("sha256") or path.stat().st_size != item.get("size"):
            raise ValueError(f"release artifact does not match receipt: {path.name}")
    return receipt


def _authenticated_request(url: str) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    username = os.environ.get("UV_INDEX_USERNAME")
    password = os.environ.get("UV_INDEX_PASSWORD")
    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    return request


def verify_index_receipt(receipt_path: Path, simple_base_url: str) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    packages = receipt.get("packages")
    artifacts = receipt.get("artifacts")
    if not isinstance(packages, list) or not isinstance(artifacts, list):
        raise ValueError("release receipt packages or artifacts are invalid")
    links: dict[str, str] = {}
    base = simple_base_url.rstrip("/") + "/"
    for package in packages:
        if not isinstance(package, str):
            raise ValueError("release receipt package name is invalid")
        normalized = re.sub(r"[-_.]+", "-", package).lower()
        page_url = urljoin(base, quote(normalized) + "/")
        try:
            with urllib.request.urlopen(_authenticated_request(page_url), timeout=20) as response:  # noqa: S310
                document = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            raise RuntimeError(f"cannot read package index page for {package}") from error
        parser = _Links()
        parser.feed(document)
        links.update({name: urljoin(page_url, href) for name, href in parser.links.items()})
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ValueError("release receipt artifact is invalid")
        filename = item["filename"]
        url = links.get(filename)
        if url is None:
            raise ValueError(f"index is missing receipt artifact: {filename}")
        try:
            with urllib.request.urlopen(_authenticated_request(url), timeout=60) as response:  # noqa: S310
                digest = hashlib.sha256(response.read()).hexdigest()
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            raise RuntimeError(f"cannot read index artifact: {filename}") from error
        if digest != item.get("sha256"):
            raise ValueError(f"index artifact hash does not match receipt: {filename}")


__all__ = [
    "create_distribution_receipt",
    "verify_distribution_receipt",
    "verify_index_receipt",
    "write_distribution_receipt",
]
