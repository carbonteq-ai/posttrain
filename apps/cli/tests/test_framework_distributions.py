"""What gets staged into an actual-job image when there is no checkout."""

from __future__ import annotations

from posttrain_cli.framework_distributions import (
    FRAMEWORK_DISTRIBUTIONS,
    IMAGE_ONLY_DISTRIBUTION,
    _index_arguments,
    installed_versions,
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


def test_a_uv_only_consumer_still_has_their_index_used() -> None:
    """pip reads PIP_INDEX_URL; the documented setup only sets the uv variable."""
    index = "https://pypi.example.invalid/simple/"
    assert _index_arguments({"UV_INDEX_URL": index}) == ["--index-url", index]
    assert _index_arguments({"UV_DEFAULT_INDEX": index}) == ["--index-url", index]
    assert _index_arguments({}) == []


def test_an_explicit_pip_index_is_left_alone() -> None:
    """A consumer who configured pip directly has already said what they want."""
    assert _index_arguments({"PIP_INDEX_URL": "https://pip.example.invalid/simple/"}) == []
