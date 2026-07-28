"""Add an operator's certificate authorities to the ones the image already has.

A site with internal TLS has to tell the job about a certificate authority the
image has never heard of. Pointing `SSL_CERT_FILE` straight at that authority
is the obvious way to do it and is wrong: the variable replaces the trust store
rather than extending it, so a job that gains its own registry loses every
public authority at the same time. The symptom appears much later and somewhere
else, as a model download failing to verify huggingface.co.

Only the container knows what the image already trusted, so the union is built
here rather than by whichever provider mounted the file.
"""

from __future__ import annotations

import os
import ssl
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

EXTRA_BUNDLE_VARIABLE = "POSTTRAIN_EXTRA_CA_BUNDLE"
"""Names a file of additional authorities to trust, never a replacement set."""

# Python's ssl module reads the first, requests and huggingface_hub the second,
# and anything the job shells out to reads one of the last two. All four have
# replace semantics, which is why only the merged bundle is ever named here.
_APPLIED_VARIABLES = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "GIT_SSL_CAINFO",
)

_BEGIN = "-----BEGIN CERTIFICATE-----"


def _image_bundles() -> list[Path]:
    """Every certificate store the image itself already provides."""
    candidates: list[str | None] = []
    try:
        import certifi

        candidates.append(certifi.where())
    except ImportError:
        pass
    verify_paths = ssl.get_default_verify_paths()
    candidates.extend((verify_paths.cafile, verify_paths.openssl_cafile))
    candidates.append("/etc/ssl/certs/ca-certificates.crt")

    found: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and not any(path.samefile(seen) for seen in found):
            found.append(path)
    return found


def _certificates(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return []
    blocks = text.split(_BEGIN)[1:]
    return [f"{_BEGIN}{block.rstrip()}\n" for block in blocks]


def install_additional_trust(environ: MutableMapping[str, str] | None = None) -> Path | None:
    """Merge the operator's authorities with the image's and publish the result.

    Returns the merged bundle, or None when the operator supplied none. The
    merge is skipped rather than guessed if the extra file holds no
    certificates, because trusting nothing new is better than replacing
    everything with nothing.
    """
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    configured = target.get(EXTRA_BUNDLE_VARIABLE, "").strip()
    if not configured:
        return None
    extra = Path(configured)
    additional = _certificates(extra)
    if not additional:
        raise RuntimeError(f"{EXTRA_BUNDLE_VARIABLE} names no certificates: {extra}")

    seen: set[str] = set()
    merged: list[str] = []
    for source in (*_image_bundles(), extra):
        for certificate in _certificates(source):
            if certificate not in seen:
                seen.add(certificate)
                merged.append(certificate)

    destination = Path(tempfile.gettempdir()) / "posttrain-ca-certificates.crt"
    destination.write_text("".join(merged), encoding="utf-8")
    destination.chmod(0o644)
    for name in _APPLIED_VARIABLES:
        target[name] = str(destination)
    return destination


__all__ = ["EXTRA_BUNDLE_VARIABLE", "install_additional_trust"]
