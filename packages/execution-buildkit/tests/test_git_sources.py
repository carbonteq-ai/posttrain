from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain_execution_buildkit import (
    GitSourceRequest,
    ImmutableGitSourcePacker,
)

_REPOSITORY = "https://github.com/PrimeIntellect-ai/verifiers"
_REVISION = "a" * 40


class FakeGitGateway:
    def __init__(
        self,
        trees: Mapping[tuple[str, str], Mapping[str, str | tuple[str, str]]],
    ) -> None:
        self.trees = trees
        self.calls: list[tuple[str, ...]] = []
        self.repositories: dict[Path, str] = {}
        self.revisions: dict[Path, str] = {}
        self.head_overrides: dict[Path, str] = {}
        self.status_overrides: dict[Path, str] = {}

    def invoke(self, arguments: Sequence[str]) -> str:
        call = tuple(arguments)
        self.calls.append(call)
        if call[:2] == ("init", "--quiet"):
            root = Path(call[2])
            (root / ".git").mkdir(parents=True)
            return ""

        assert call[0] == "-C"
        root = Path(call[1])
        command = call[2]
        if call[2:5] == ("remote", "add", "origin"):
            self.repositories[root] = call[5]
            return ""
        if command == "fetch":
            self.revisions[root] = call[-1]
            return ""
        if command == "checkout":
            repository = self.repositories[root]
            revision = self.revisions[root]
            for relative, content in self.trees[(repository, revision)].items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, tuple):
                    kind, value = content
                    assert kind == "symlink"
                    target.symlink_to(value)
                else:
                    target.write_text(content)
            return ""
        if command == "rev-parse":
            default_revision = self.revisions.get(root)
            if default_revision is None:
                known_revisions = set(self.revisions.values())
                assert len(known_revisions) == 1
                default_revision = known_revisions.pop()
            return self.head_overrides.get(root, default_revision) + "\n"
        if command == "status":
            return self.status_overrides.get(root, "")
        raise AssertionError(f"unexpected Git invocation: {call}")


def _gateway() -> FakeGitGateway:
    return FakeGitGateway(
        {
            (_REPOSITORY, _REVISION): {
                "README.md": "verifiers",
                "environments/gsm8k/pyproject.toml": "[project]\nname='gsm8k'",
                "environments/gsm8k/gsm8k/__init__.py": "def load(): ...",
                "environments/reverse_text/pyproject.toml": ("[project]\nname='reverse-text'"),
            }
        }
    )


def test_materializes_deduplicated_sources_and_emits_deterministic_lock(
    tmp_path: Path,
) -> None:
    gateway = _gateway()
    packer = ImmutableGitSourcePacker(
        cache_root=(tmp_path / "cache").absolute(),
        gateway=gateway,
    )
    requests = [
        GitSourceRequest(
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectories=("environments/reverse_text",),
        ),
        GitSourceRequest(
            repository=_REPOSITORY,
            revision=_REVISION,
            subdirectories=("environments/gsm8k",),
        ),
    ]

    result = packer.materialize(requests)
    repeated = packer.materialize(list(reversed(requests)))

    assert len(result.sources) == 1
    assert result.lock == repeated.lock
    assert result.lock.digest == repeated.lock.digest
    assert [item.path for item in result.lock.sources[0].subdirectories] == [
        "environments/gsm8k",
        "environments/reverse_text",
    ]
    assert len(result.lock.sources[0].source_tree_digest) == 64
    assert all(len(item.tree_digest) == 64 for item in result.lock.sources[0].subdirectories)
    assert str(tmp_path) not in result.lock.to_json()
    assert result.lock.as_dict()["schema"] == "posttrain.git-source-lock.v1"
    assert sum(call[2] == "fetch" for call in gateway.calls if call[0] == "-C") == 1
    assert all(isinstance(call, tuple) for call in gateway.calls)


@pytest.mark.parametrize(
    ("repository", "revision", "subdirectories"),
    [
        ("https://user:secret@github.com/org/repo", _REVISION, ("env",)),
        ("https://github.com/org/repo?token=secret", _REVISION, ("env",)),
        ("https://github.com/org/repo/", _REVISION, ("env",)),
        ("https://GitHub.com/org/repo", _REVISION, ("env",)),
        (_REPOSITORY, "a" * 39, ("env",)),
        (_REPOSITORY, _REVISION.upper(), ("env",)),
        (_REPOSITORY, _REVISION, ("../env",)),
        (_REPOSITORY, _REVISION, ("env/./nested",)),
        (_REPOSITORY, _REVISION, ("/env",)),
        (_REPOSITORY, _REVISION, ("env", "env")),
        (_REPOSITORY, _REVISION, ()),
    ],
)
def test_rejects_noncanonical_or_mutable_source_identity(
    repository: str,
    revision: str,
    subdirectories: tuple[str, ...],
) -> None:
    with pytest.raises(ContractError):
        GitSourceRequest(
            repository=repository,
            revision=revision,
            subdirectories=subdirectories,
        )


def test_rejects_wrong_head_before_publishing_cache_entry(tmp_path: Path) -> None:
    gateway = _gateway()
    packer = ImmutableGitSourcePacker(
        cache_root=(tmp_path / "cache").absolute(),
        gateway=gateway,
    )
    original_invoke = gateway.invoke

    def wrong_head(arguments: Sequence[str]) -> str:
        if len(arguments) > 2 and arguments[2] == "rev-parse":
            return "b" * 40
        return original_invoke(arguments)

    gateway.invoke = wrong_head  # type: ignore[method-assign]

    with pytest.raises(ContractError, match="HEAD mismatch"):
        packer.materialize(
            [
                GitSourceRequest(
                    repository=_REPOSITORY,
                    revision=_REVISION,
                    subdirectories=("environments/gsm8k",),
                )
            ]
        )

    assert list((tmp_path / "cache").iterdir()) == []


def test_rejects_dirty_cached_checkout(tmp_path: Path) -> None:
    gateway = _gateway()
    packer = ImmutableGitSourcePacker(
        cache_root=(tmp_path / "cache").absolute(),
        gateway=gateway,
    )
    request = GitSourceRequest(
        repository=_REPOSITORY,
        revision=_REVISION,
        subdirectories=("environments/gsm8k",),
    )
    result = packer.materialize([request])
    root = result.sources[0].root
    gateway.status_overrides[root] = " M README.md\n"

    with pytest.raises(ContractError, match="dirty filesystem drift"):
        packer.materialize([request])


def test_rejects_symlinks_and_missing_subdirectories(tmp_path: Path) -> None:
    symlink_gateway = FakeGitGateway(
        {
            (_REPOSITORY, _REVISION): {
                "environment/pyproject.toml": "[project]",
                "environment/escape": ("symlink", "../../outside"),
            }
        }
    )
    request = GitSourceRequest(
        repository=_REPOSITORY,
        revision=_REVISION,
        subdirectories=("environment",),
    )

    with pytest.raises(ContractError, match="do not accept symlinks"):
        ImmutableGitSourcePacker(
            cache_root=(tmp_path / "symlink-cache").absolute(),
            gateway=symlink_gateway,
        ).materialize([request])

    with pytest.raises(ContractError, match="does not exist"):
        ImmutableGitSourcePacker(
            cache_root=(tmp_path / "missing-cache").absolute(),
            gateway=_gateway(),
        ).materialize(
            [
                GitSourceRequest(
                    repository=_REPOSITORY,
                    revision=_REVISION,
                    subdirectories=("environments/missing",),
                )
            ]
        )


def test_requires_absolute_cache_root(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="absolute"):
        ImmutableGitSourcePacker(cache_root=Path("relative"), gateway=_gateway())


def test_rejects_two_revisions_of_one_repository_before_fetch(tmp_path: Path) -> None:
    gateway = _gateway()
    packer = ImmutableGitSourcePacker(
        cache_root=(tmp_path / "cache").absolute(),
        gateway=gateway,
    )

    with pytest.raises(ContractError, match="multiple revisions"):
        packer.materialize(
            [
                GitSourceRequest(
                    repository=_REPOSITORY,
                    revision=_REVISION,
                    subdirectories=("environments/gsm8k",),
                ),
                GitSourceRequest(
                    repository=_REPOSITORY,
                    revision="b" * 40,
                    subdirectories=("environments/reverse_text",),
                ),
            ]
        )

    assert gateway.calls == []


def test_rejects_unlocked_submodules(tmp_path: Path) -> None:
    gateway = FakeGitGateway(
        {
            (_REPOSITORY, _REVISION): {
                ".gitmodules": ('[submodule "shared"]\n\tpath = shared\n\turl = https://github.com/example/shared\n'),
                "environment/pyproject.toml": "[project]",
            }
        }
    )

    with pytest.raises(ContractError, match="submodule lock"):
        ImmutableGitSourcePacker(
            cache_root=(tmp_path / "cache").absolute(),
            gateway=gateway,
        ).materialize(
            [
                GitSourceRequest(
                    repository=_REPOSITORY,
                    revision=_REVISION,
                    subdirectories=("environment",),
                )
            ]
        )
