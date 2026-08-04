"""Immutable, idempotent ReViSQL and BIRD asset preparation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_ENV = "POSTTRAIN_SKYRL_BIRD_CACHE"
REVISQL_REVISION = "9fac371aa22019e9912dcbd6572e8fe8194d352a"
BIRD_REVISION = "7877a1bfee6b3794f5026b1f00fcc4dd43e529be"
TRAIN_SHA256 = "a3829a81a02299e3a0155afa1321e1a4cca58ba90645bb95a59e6b8a1de8b3ec"
VALIDATION_SHA256 = "a2781fab072244928d4d4e452b626d7ee2133050353b0bba6c631159d99ed39e"
DATABASE_ARCHIVE_SHA256 = "54424b2004cea43f1fd89605b3df41836df3a46bc68ffd5444c6549c112172f3"
DATABASE_ARCHIVE_SIZE = 20_683_638_742
TRAIN_ROWS = 2_064
VALIDATION_ROWS = 398

_TRAIN_URL = (
    "https://raw.githubusercontent.com/uiuc-kang-lab/ReViSQL/"
    f"{REVISQL_REVISION}/data/bird-verified-train.json"
)
_VALIDATION_URL = (
    "https://raw.githubusercontent.com/uiuc-kang-lab/ReViSQL/"
    f"{REVISQL_REVISION}/data/bird-verified-val.json"
)
_DATABASE_URL = (
    "https://huggingface.co/datasets/Sudnya/bird-sql/resolve/"
    f"{BIRD_REVISION}/databases/train_databases.zip?download=true"
)


@dataclass(frozen=True, slots=True)
class AssetLayout:
    root: Path

    @property
    def train_json(self) -> Path:
        return self.root / "revisql" / "bird-verified-train.json"

    @property
    def validation_json(self) -> Path:
        return self.root / "revisql" / "bird-verified-val.json"

    @property
    def archive(self) -> Path:
        return self.root / "downloads" / "train_databases.zip"

    @property
    def databases(self) -> Path:
        return self.root / "databases"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def cache_layout(root: Path | None = None) -> AssetLayout:
    selected = root
    if selected is None:
        raw = os.environ.get(CACHE_ENV)
        if not raw:
            raise RuntimeError(f"set {CACHE_ENV} to a persistent asset directory")
        selected = Path(raw)
    return AssetLayout(selected.expanduser().resolve())


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "posttrain-skyrl-bird-sql/1"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _ensure_download(url: str, path: Path, digest: str, size: int | None = None) -> None:
    if path.is_file() and (size is None or path.stat().st_size == size) and sha256_file(path) == digest:
        return
    path.unlink(missing_ok=True)
    _download(url, path)
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"downloaded {path.name} has size {path.stat().st_size}, expected {size}")
    actual = sha256_file(path)
    if actual != digest:
        raise RuntimeError(f"downloaded {path.name} has sha256 {actual}, expected {digest}")


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            candidate = (root / member.filename).resolve()
            if candidate != root and root not in candidate.parents:
                raise RuntimeError(f"unsafe archive path: {member.filename}")
            mode = member.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise RuntimeError(f"archive symlinks are not allowed: {member.filename}")
        bundle.extractall(root)


def _database_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted((*root.rglob("*.sqlite"), *root.rglob("*.db"))):
        db_id = path.parent.name
        if db_id in index and index[db_id] != path:
            raise RuntimeError(f"database id {db_id!r} resolves to multiple files")
        index[db_id] = path
    return index


def database_path(db_id: str, root: Path | None = None) -> Path:
    layout = cache_layout(root)
    try:
        return _database_index(layout.databases)[db_id]
    except KeyError as error:
        raise FileNotFoundError(f"BIRD database {db_id!r} is not prepared under {layout.databases}") from error


def load_rows(split: str, root: Path | None = None) -> list[dict[str, Any]]:
    layout = cache_layout(root)
    path = layout.train_json if split == "train" else layout.validation_json
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise RuntimeError(f"{path} must contain a JSON array of objects")
    return payload


@contextmanager
def _preparation_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def prepare(root: Path | None = None) -> dict[str, Any]:
    layout = cache_layout(root)
    with _preparation_lock(layout.root):
        _ensure_download(_TRAIN_URL, layout.train_json, TRAIN_SHA256)
        _ensure_download(_VALIDATION_URL, layout.validation_json, VALIDATION_SHA256)
        _ensure_download(
            _DATABASE_URL,
            layout.archive,
            DATABASE_ARCHIVE_SHA256,
            DATABASE_ARCHIVE_SIZE,
        )
        required_databases = {
            str(row["db_id"])
            for row in (*load_rows("train", layout.root), *load_rows("validation", layout.root))
        }
        prepared_databases = _database_index(layout.databases) if layout.databases.is_dir() else {}
        if not required_databases.issubset(prepared_databases):
            with tempfile.TemporaryDirectory(prefix="bird-databases-", dir=layout.root) as temporary:
                staging = Path(temporary) / "databases"
                staging.mkdir()
                _safe_extract(layout.archive, staging)
                staged_databases = _database_index(staging)
                missing = required_databases - staged_databases.keys()
                if missing:
                    raise RuntimeError(f"the BIRD archive is missing {len(missing)} required databases")
                if layout.databases.exists():
                    shutil.rmtree(layout.databases)
                staging.replace(layout.databases)
        report = validate(layout.root, require_manifest=False)
        temporary_manifest = layout.manifest.with_suffix(".json.partial")
        temporary_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_manifest.replace(layout.manifest)
        return report


def validate(root: Path | None = None, *, require_manifest: bool = True) -> dict[str, Any]:
    layout = cache_layout(root)
    checks = (
        (layout.train_json, TRAIN_SHA256),
        (layout.validation_json, VALIDATION_SHA256),
        (layout.archive, DATABASE_ARCHIVE_SHA256),
    )
    for path, expected in checks:
        if not path.is_file():
            raise RuntimeError(f"missing prepared asset: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{path} has sha256 {actual}, expected {expected}")
    if layout.archive.stat().st_size != DATABASE_ARCHIVE_SIZE:
        raise RuntimeError("the BIRD database archive size does not match the pinned artifact")

    train = load_rows("train", layout.root)
    validation = load_rows("validation", layout.root)
    if len(train) != TRAIN_ROWS or len(validation) != VALIDATION_ROWS:
        raise RuntimeError(
            f"unexpected ReViSQL row counts: train={len(train)}, validation={len(validation)}"
        )
    train_ids = {str(row["question_id"]) for row in train}
    validation_ids = {str(row["question_id"]) for row in validation}
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"train/validation question IDs overlap: {len(overlap)}")
    databases = _database_index(layout.databases)
    required = {str(row["db_id"]) for row in (*train, *validation)}
    missing = required - databases.keys()
    if missing:
        raise RuntimeError(f"prepared archive is missing {len(missing)} required databases")
    report: dict[str, Any] = {
        "schema": "posttrain.skyrl-bird-sql-assets.v1",
        "revisql_revision": REVISQL_REVISION,
        "bird_revision": BIRD_REVISION,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_sha256": TRAIN_SHA256,
        "validation_sha256": VALIDATION_SHA256,
        "database_archive_sha256": DATABASE_ARCHIVE_SHA256,
        "database_archive_size": DATABASE_ARCHIVE_SIZE,
        "required_database_count": len(required),
        "prepared_database_count": len(databases),
        "question_id_overlap": 0,
    }
    if require_manifest:
        if not layout.manifest.is_file():
            raise RuntimeError("asset manifest is missing; run prepare")
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        if manifest != report:
            raise RuntimeError("asset manifest does not match the prepared assets")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(prog="python -m skyrl_bird_sql_v1.assets")
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    report = prepare(args.root) if args.command == "prepare" else validate(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()


__all__ = [
    "AssetLayout",
    "cache_layout",
    "database_path",
    "load_rows",
    "prepare",
    "sha256_file",
    "validate",
]
