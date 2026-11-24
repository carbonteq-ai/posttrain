"""p9_adapter_check.py: the lab's tree digest, and the check passes and fails both ways."""
import hashlib

from p9_adapter_check import check, tree_sha256


def make(folder, weights=b"weights", extra=True):
    folder.mkdir()
    (folder / "adapter_model.safetensors").write_bytes(weights)
    (folder / "adapter_config.json").write_bytes(b"{}")
    if extra:
        (folder / "tokenizer.json").write_bytes(b"tok")
    return folder


def test_tree_digest_matches_the_lab_definition(tmp_path):
    run = make(tmp_path / "run")
    want = hashlib.sha256()
    for name in sorted(["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"]):
        want.update(name.encode() + b"\0" + (run / name).read_bytes())
    assert tree_sha256(run) == want.hexdigest()


def test_check_passes_for_the_same_adapter_and_fails_for_another(tmp_path):
    run = make(tmp_path / "run")
    same = make(tmp_path / "evaluated", extra=False)
    other = make(tmp_path / "other", weights=b"different", extra=False)
    digest = tree_sha256(run)
    assert check(run, same, digest)["ok"]
    assert not check(run, other, digest)["ok"]
    assert not check(run, same, "0" * 64)["ok"]


def test_tree_digest_equals_the_lab_function(tmp_path):
    import pytest

    lab = pytest.importorskip("posttrain_lab.tracking.trackio_observer")
    run = make(tmp_path / "run")
    assert tree_sha256(run) == lab._tree_sha256(run)
