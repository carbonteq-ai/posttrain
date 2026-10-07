"""A live run's rollout population refreshes from its newest end."""

from __future__ import annotations

import pytest
from posttrain.tracking import TracePage, TraceQuery, TraceRecord
from posttrain_observatory.traces import newer_trace_summaries


class _PagedTraces:
    """Newest-first trace pages over a list that grows at its newest end."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.reads: list[tuple[str | None, int]] = []

    async def traces(self, run_id: str, query: TraceQuery) -> TracePage:
        assert query.order == "newest_first"
        offset = int(query.cursor or 0)
        self.reads.append((query.cursor, query.limit))
        newest_first = list(reversed(self.ids))
        items = newest_first[offset : offset + query.limit]
        following = offset + query.limit
        return TracePage(
            items=tuple(
                TraceRecord(trace_type="verifiers", external_id=item, payload={"reward": 1.0}) for item in items
            ),
            next_cursor=str(following) if following < len(newest_first) else None,
            live=True,
        )


@pytest.mark.asyncio
async def test_refresh_reads_only_traces_newer_than_the_cached_population() -> None:
    source = _PagedTraces([f"t{index:03d}" for index in range(500)])
    known = frozenset(source.ids)
    source.ids += [f"t{index:03d}" for index in range(500, 530)]

    newer, live = await newer_trace_summaries(
        source,  # type: ignore[arg-type]
        "run",
        trace_type="verifiers",
        metadata=None,
        known=known,
        page_size=20,
    )

    assert [item.external_id for item in newer] == [f"t{index:03d}" for index in range(529, 499, -1)]
    assert live is True
    # Two pages held new traces; the third held none and ended the read, far short of 27 pages.
    assert len(source.reads) == 3


@pytest.mark.asyncio
async def test_refresh_with_nothing_new_reads_one_page() -> None:
    source = _PagedTraces([f"t{index}" for index in range(100)])
    newer, _ = await newer_trace_summaries(
        source,  # type: ignore[arg-type]
        "run",
        trace_type="verifiers",
        metadata=None,
        known=frozenset(source.ids),
        page_size=20,
    )
    assert newer == ()
    assert len(source.reads) == 1
