"""Adding an internal authority must not remove the public ones."""

from __future__ import annotations

import ssl
from pathlib import Path

import pytest
from posttrain_runtime.trust import EXTRA_BUNDLE_VARIABLE, install_additional_trust


def _certificate(label: str) -> str:
    return f"-----BEGIN CERTIFICATE-----\n{label}\n-----END CERTIFICATE-----\n"


def test_nothing_changes_when_no_extra_authority_is_configured() -> None:
    environ: dict[str, str] = {}

    assert install_additional_trust(environ) is None
    assert environ == {}


def test_the_internal_authority_is_added_to_the_image_s_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this prevents surfaces nowhere near its cause.

    Pointing SSL_CERT_FILE at an internal authority replaces the trust store
    rather than extending it, so a job that gains its own registry loses every
    public authority at the same moment. It then fails verifying huggingface.co
    while downloading a model, which reads as a broken model reference.
    """
    image_bundle = tmp_path / "image-ca.crt"
    image_bundle.write_text(_certificate("PUBLIC-ROOT"), encoding="utf-8")
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: ssl.DefaultVerifyPaths(str(image_bundle), "", "", str(image_bundle), "", ""),
    )
    internal = tmp_path / "internal-ca.crt"
    internal.write_text(_certificate("INTERNAL-ROOT"), encoding="utf-8")
    environ = {EXTRA_BUNDLE_VARIABLE: str(internal)}

    merged = install_additional_trust(environ)

    assert merged is not None
    text = merged.read_text(encoding="utf-8")
    assert "INTERNAL-ROOT" in text
    assert "PUBLIC-ROOT" in text, "the image's own authorities must survive"
    assert environ["SSL_CERT_FILE"] == str(merged)
    assert environ["REQUESTS_CA_BUNDLE"] == str(merged)


def test_an_authority_present_in_both_is_written_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _certificate("SHARED-ROOT")
    image_bundle = tmp_path / "image-ca.crt"
    image_bundle.write_text(shared, encoding="utf-8")
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: ssl.DefaultVerifyPaths(str(image_bundle), "", "", str(image_bundle), "", ""),
    )
    internal = tmp_path / "internal-ca.crt"
    internal.write_text(shared, encoding="utf-8")

    merged = install_additional_trust({EXTRA_BUNDLE_VARIABLE: str(internal)})

    assert merged is not None
    assert merged.read_text(encoding="utf-8").count("SHARED-ROOT") == 1


def test_a_bundle_holding_no_certificates_is_refused(tmp_path: Path) -> None:
    """Replacing the trust store with nothing is worse than trusting nothing new."""
    empty = tmp_path / "empty.crt"
    empty.write_text("# no certificates here\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="names no certificates"):
        install_additional_trust({EXTRA_BUNDLE_VARIABLE: str(empty)})
