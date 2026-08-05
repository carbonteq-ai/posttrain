"""Allocate immutable release-candidate versions from the development index."""

from __future__ import annotations

import base64
import os
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse

_FINAL_VERSION = re.compile(r"^[0-9]+[.][0-9]+[.][0-9]+$")


class _SimpleLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = next((value for name, value in attrs if name == "href"), None)
        if href:
            self.links.append(unquote(urlparse(href).path.rsplit("/", 1)[-1]))


def next_candidate_version(target: str, artifacts: tuple[str, ...]) -> str:
    """Return the first RC number not represented by a Posttrain artifact."""

    if not _FINAL_VERSION.fullmatch(target):
        raise ValueError(f"candidate target must be a final X.Y.Z version: {target!r}")
    pattern = re.compile(
        rf"^posttrain[-_]{re.escape(target)}rc(?P<number>[1-9][0-9]*)(?:[-_.]|$)",
        re.IGNORECASE,
    )
    allocated = {
        int(match.group("number")) for artifact in artifacts if (match := pattern.search(artifact)) is not None
    }
    number = 1
    while number in allocated:
        number += 1
    return f"{target}rc{number}"


def fetch_simple_artifacts(
    simple_url: str,
    *,
    username: str | None = None,
    password: str | None = None,
) -> tuple[str, ...]:
    """Read filenames from one PEP 503 project page without exposing credentials."""

    request = urllib.request.Request(simple_url, headers={"Accept": "text/html"})
    resolved_username = username or os.environ.get("UV_INDEX_USERNAME")
    resolved_password = password or os.environ.get("UV_INDEX_PASSWORD")
    if resolved_username and resolved_password:
        token = base64.b64encode(f"{resolved_username}:{resolved_password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - reviewed HTTPS index URL
            document = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return ()
        raise RuntimeError(f"development index returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot read development index: {error.reason}") from error
    parser = _SimpleLinks()
    parser.feed(document)
    return tuple(parser.links)


__all__ = ["fetch_simple_artifacts", "next_candidate_version"]
