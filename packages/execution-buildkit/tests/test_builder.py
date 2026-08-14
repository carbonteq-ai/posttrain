from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import RuntimeImageRef
from posttrain_execution_buildkit import (
    BuildKitRuntimeBuilder,
    BuildxCli,
    RemoteImageNotFoundError,
    RuntimeBuildRequest,
    RuntimeImageInspector,
    digest_runtime_sources,
)


class _Process:
    def __init__(self, output: str, returncode: int = 0) -> None:
        self.stdout = io.StringIO(output)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def test_buildx_cli_keeps_image_inspection_json_off_the_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "posttrain_execution_buildkit.builder.subprocess.Popen",
        lambda *args, **kwargs: _Process('{"config":{"Labels":{}}}\n'),
    )

    output = BuildxCli().invoke(("imagetools", "inspect", "registry.lan/example", "--format", "{{json .Image}}"))

    assert output == '{"config":{"Labels":{}}}\n'
    assert capsys.readouterr().err == ""


def test_buildx_cli_streams_bake_progress(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "posttrain_execution_buildkit.builder.subprocess.Popen",
        lambda *args, **kwargs: _Process("#1 loading build definition\n"),
    )

    output = BuildxCli().invoke(("bake", "--progress=plain", "posttrain-job"))

    assert output == "#1 loading build definition\n"
    assert capsys.readouterr().err == "#1 loading build definition\n"


def test_image_copy_reuses_a_matching_destination_without_creating_it() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def invoke(self, arguments):
            call = tuple(arguments)
            self.calls.append(call)
            if "{{json .Manifest.Digest}}" in call:
                return json.dumps("sha256:" + "a" * 64)
            if "{{json .Image}}" in call:
                return json.dumps({"config": {"Labels": {}}})
            raise AssertionError(call)

    gateway = Gateway()
    inspector = RuntimeImageInspector(gateway)

    digest, copied = inspector.ensure_copy(
        "source.example/kind@sha256:" + "a" * 64,
        "destination.example/kind:v1",
        expected_digest="sha256:" + "a" * 64,
    )

    assert digest == "sha256:" + "a" * 64
    assert not copied
    assert not any(call[:2] == ("imagetools", "create") for call in gateway.calls)


def test_image_copy_populates_a_missing_destination_once() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.available = False

        def invoke(self, arguments):
            call = tuple(arguments)
            self.calls.append(call)
            if call[:2] == ("imagetools", "create"):
                self.available = True
                return ""
            if "{{json .Manifest.Digest}}" in call:
                if not self.available:
                    raise RemoteImageNotFoundError("manifest unknown")
                return json.dumps("sha256:" + "a" * 64)
            if "{{json .Image}}" in call:
                return json.dumps({"config": {"Labels": {}}})
            raise AssertionError(call)

    gateway = Gateway()
    inspector = RuntimeImageInspector(gateway)

    digest, copied = inspector.ensure_copy(
        "source.example/kind@sha256:" + "a" * 64,
        "destination.example/kind:v1",
        expected_digest="sha256:" + "a" * 64,
    )

    assert digest == "sha256:" + "a" * 64
    assert copied
    assert sum(call[:2] == ("imagetools", "create") for call in gateway.calls) == 1


def test_runtime_builder_copies_and_rechecks_an_immutable_digest(tmp_path: Path) -> None:
    class Gateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def invoke(self, arguments):
            call = tuple(arguments)
            self.calls.append(call)
            if call[:2] == ("imagetools", "create"):
                return ""
            if call[:2] == ("imagetools", "inspect"):
                return json.dumps("sha256:" + "a" * 64)
            raise AssertionError(call)

    gateway = Gateway()
    builder = BuildKitRuntimeBuilder(gateway, receipt_root=(tmp_path / "receipts").resolve())
    copied = builder.copy_remote(
        RuntimeImageRef("source.example/posttrain-base@sha256:" + "a" * 64),
        destination_repository="destination.example/posttrain-base",
        expected_digest="sha256:" + "a" * 64,
        tag="reuse-0123456789abcdef",
    )

    assert copied.value == "destination.example/posttrain-base@sha256:" + "a" * 64
    assert gateway.calls[0] == (
        "imagetools",
        "create",
        "--tag",
        "destination.example/posttrain-base:reuse-0123456789abcdef",
        "source.example/posttrain-base@sha256:" + "a" * 64,
    )
    assert gateway.calls[1][:2] == ("imagetools", "inspect")


class FakeBuildx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.digest = "d" * 64

    def invoke(self, arguments):
        call = tuple(arguments)
        self.calls.append(call)
        if "--metadata-file" in call:
            path = Path(call[call.index("--metadata-file") + 1])
            target = call[-1]
            path.write_text(
                json.dumps(
                    {
                        target: {
                            "containerimage.digest": f"sha256:{self.digest}",
                        }
                    }
                ),
                encoding="utf-8",
            )
            return ""
        if call[:2] == ("imagetools", "inspect"):
            return json.dumps(f"sha256:{self.digest}")
        return ""


class TransientUploadBuildx(FakeBuildx):
    def __init__(self, *, always_fail: bool = False, failure: str = "invalid content range") -> None:
        super().__init__()
        self.always_fail = always_fail
        self.failure = failure
        self.failed_builds = 0

    def invoke(self, arguments):
        call = tuple(arguments)
        if "--metadata-file" in call and (self.always_fail or self.failed_builds == 0):
            self.calls.append(call)
            self.failed_builds += 1
            raise RuntimeError(f"failed to push image: unknown: {self.failure}")
        return super().invoke(arguments)


def _request(tmp_path: Path) -> RuntimeBuildRequest:
    bake = tmp_path / "docker-bake.hcl"
    bake.write_text('target "training-verl" {}\n', encoding="utf-8")
    return RuntimeBuildRequest(
        profile="training/verl@1",
        bake_file=bake.resolve(),
        context=tmp_path.resolve(),
        target="training-verl",
        repository="registry.lan/carbonteq/posttrain",
        source_digest="a" * 64,
        lock_digest="b" * 64,
        base_image=RuntimeImageRef(f"registry.lan/base@sha256:{'c' * 64}"),
        builder="posttrain-builder",
        variables={"FRAMEWORK_VERSION": "0.1.0"},
    )


def test_default_push_skips_attestations(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    builder = BuildKitRuntimeBuilder(
        gateway,
        receipt_root=(tmp_path / "receipts").resolve(),
    )
    request = _request(tmp_path)
    builder.build(request)
    build_calls = [call for call in gateway.calls if "--metadata-file" in call]
    call = build_calls[0]
    assert ("--provenance", "false") == (call[call.index("--provenance")], call[call.index("--provenance") + 1])
    assert ("--sbom", "false") == (call[call.index("--sbom")], call[call.index("--sbom") + 1])
    assert any("compression-level=1" in item for item in call)
    assert any("force-compression=false" in item for item in call)


def test_attestations_opt_in_and_change_the_build_key(tmp_path: Path) -> None:
    request = _request(tmp_path)
    attested = replace(request, attestations=True, compression_level=3, force_compression=True)
    assert request.build_key != attested.build_key
    gateway = FakeBuildx()
    builder = BuildKitRuntimeBuilder(gateway, receipt_root=(tmp_path / "receipts").resolve())
    builder.build(attested)
    call = [c for c in gateway.calls if "--metadata-file" in c][0]
    assert ("--provenance", "mode=max") == (call[call.index("--provenance")], call[call.index("--provenance") + 1])


def test_builder_checks_pushes_verifies_and_reuses_receipt(tmp_path: Path) -> None:
    gateway = FakeBuildx()
    builder = BuildKitRuntimeBuilder(
        gateway,
        receipt_root=(tmp_path / "receipts").resolve(),
    )
    request = _request(tmp_path)

    first = builder.build(request)
    second = builder.build(request)

    assert first == second
    assert first.image.value == f"{request.repository}@sha256:{gateway.digest}"
    assert first.receipt.stat().st_mode & 0o777 == 0o600
    build_calls = [call for call in gateway.calls if "--metadata-file" in call]
    assert len(build_calls) == 1
    assert "--push" in build_calls[0]
    check_call = gateway.calls[0]
    assert f"{request.target}.output=type=cacheonly" in check_call
    assert f"BASE_IMAGE={request.base_image.value}" in check_call
    assert ("--var", f"BASE_IMAGE={request.base_image.value}") == (
        build_calls[0][build_calls[0].index("--var")],
        build_calls[0][build_calls[0].index("--var") + 1],
    )
    assert sum(call[:2] == ("imagetools", "inspect") for call in gateway.calls) == 2


def test_builder_retries_a_transient_registry_upload_once(tmp_path: Path) -> None:
    gateway = TransientUploadBuildx()
    builder = BuildKitRuntimeBuilder(gateway, receipt_root=(tmp_path / "receipts").resolve())

    result = builder.build(_request(tmp_path))

    assert result.image.value == f"registry.lan/carbonteq/posttrain@sha256:{gateway.digest}"
    assert len([call for call in gateway.calls if "--metadata-file" in call]) == 2


def test_builder_stops_after_one_transient_registry_upload_retry(tmp_path: Path) -> None:
    gateway = TransientUploadBuildx(always_fail=True)
    builder = BuildKitRuntimeBuilder(gateway, receipt_root=(tmp_path / "receipts").resolve())

    with pytest.raises(RuntimeError, match="invalid content range"):
        builder.build(_request(tmp_path))

    assert len([call for call in gateway.calls if "--metadata-file" in call]) == 2


@pytest.mark.parametrize("failure", ["blob upload unknown", "operation timed out"])
def test_builder_retries_known_transient_delivery_failures(tmp_path: Path, failure: str) -> None:
    gateway = TransientUploadBuildx(failure=failure)
    builder = BuildKitRuntimeBuilder(gateway, receipt_root=(tmp_path / "receipts").resolve())

    result = builder.build(_request(tmp_path))

    assert result.image.value == f"registry.lan/carbonteq/posttrain@sha256:{gateway.digest}"
    assert len([call for call in gateway.calls if "--metadata-file" in call]) == 2


def test_builder_rejects_secret_build_variables(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ContractError, match="non-secret"):
        RuntimeBuildRequest(
            profile=request.profile,
            bake_file=request.bake_file,
            context=request.context,
            target=request.target,
            repository=request.repository,
            source_digest=request.source_digest,
            lock_digest=request.lock_digest,
            base_image=request.base_image,
            variables={"REGISTRY_TOKEN": "secret"},
        )


def test_runtime_build_key_excludes_local_bake_and_context_paths(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _request(first_root)
    second_bake = second_root / "docker-bake.hcl"
    second_bake.write_bytes(first.bake_file.read_bytes())
    second = replace(
        first,
        bake_file=second_bake.resolve(),
        context=second_root.resolve(),
    )

    assert first.build_key == second.build_key


def test_runtime_trust_bundle_is_mounted_and_hashed_not_path_identified(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_bundle = first_root / "ca-bundle.pem"
    second_bundle = second_root / "renamed.pem"
    first_bundle.write_text("test public CA\n", encoding="utf-8")
    second_bundle.write_bytes(first_bundle.read_bytes())
    request = replace(_request(tmp_path), trust_bundle=first_bundle.resolve())
    same_bytes = replace(request, trust_bundle=second_bundle.resolve())

    assert request.build_key == same_bytes.build_key

    gateway = FakeBuildx()
    builder = BuildKitRuntimeBuilder(gateway, receipt_root=(tmp_path / "receipts").resolve())
    builder.build(request)
    build_call = [call for call in gateway.calls if "--metadata-file" in call][0]
    assert f"fs.read={first_bundle.resolve()}" in build_call
    assert f"{request.target}.secrets=id=posttrain_ca_bundle,src={first_bundle.resolve()}" in build_call

    second_bundle.write_text("different public CA\n", encoding="utf-8")
    assert request.build_key != replace(request, trust_bundle=second_bundle.resolve()).build_key


def test_runtime_trust_bundle_must_be_an_existing_absolute_file(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="trust_bundle"):
        replace(_request(tmp_path), trust_bundle=tmp_path / "missing.pem")


def test_runtime_builder_rejects_unprotected_cached_receipt(
    tmp_path: Path,
) -> None:
    gateway = FakeBuildx()
    builder = BuildKitRuntimeBuilder(
        gateway,
        receipt_root=(tmp_path / "receipts").resolve(),
    )
    request = _request(tmp_path)
    result = builder.build(request)
    result.receipt.chmod(0o644)

    with pytest.raises(ContractError, match="group or other"):
        builder.build(request)


def test_runtime_source_digest_tracks_content_not_input_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second.py"
    first.mkdir()
    (first / "value.txt").write_text("one\n", encoding="utf-8")
    second.write_text("print('two')\n", encoding="utf-8")

    before = digest_runtime_sources(
        tmp_path.resolve(),
        (Path("first"), Path("second.py")),
    )
    reordered = digest_runtime_sources(
        tmp_path.resolve(),
        (Path("second.py"), Path("first")),
    )
    (first / "value.txt").write_text("changed\n", encoding="utf-8")
    after = digest_runtime_sources(
        tmp_path.resolve(),
        (Path("first"), Path("second.py")),
    )

    assert before == reordered
    assert before != after
