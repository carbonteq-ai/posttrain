from types import SimpleNamespace

from posttrain.train.integrations.verifiers import _native_failure_detail


def test_native_failure_detail_prefers_trace_scoring_failure() -> None:
    episode = SimpleNamespace(
        errors=[SimpleNamespace(type="EnvError", message="older environment error")],
        traces=[
            SimpleNamespace(
                last_error=SimpleNamespace(
                    type="TaskError",
                    message=(
                        "scoring: ValueError: episode judge exhausted bounded attempts; "
                        "last_error=BadRequestError: response_format is unsupported"
                    ),
                )
            )
        ],
    )

    assert _native_failure_detail(episode) == (
        "TaskError: scoring: ValueError: episode judge exhausted bounded attempts; "
        "last_error=BadRequestError: response_format is unsupported"
    )


def test_native_failure_detail_is_bounded() -> None:
    episode = SimpleNamespace(
        errors=[SimpleNamespace(type="TaskError", message="x" * 2_000)],
        traces=[],
    )

    detail = _native_failure_detail(episode)
    assert detail is not None
    assert len(detail) == 700
