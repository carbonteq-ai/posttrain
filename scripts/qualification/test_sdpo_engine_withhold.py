"""sdpo-weekend (plan v4, 14 Nov): EngineContext(withhold_untrue=True) withholds the items whose lines the truth checker
or the leak test rejects, rebuilds the context from the reduced Record and re-checks it; off, it refuses as before."""
import pytest

import sdpo_ab_engine as engine


class _Item:
    def __init__(self, i):
        self.id, self.origin = i, "native"


class _Record:
    def __init__(self, ids):
        self.items, self.notes = tuple(_Item(i) for i in ids), []

    def model_copy(self, update):
        r = _Record([]); r.items = update["items"]; return r


class _Episode:
    def __init__(self, record):
        self.id, self.task, self.success, self.messages = record["id"], "t", False, []


def _engine(monkeypatch, withhold, check):
    built = []

    def build_context(record, others, **kw):
        built.append([i.id for i in record.items])
        return {"first_message": " ".join(i.id for i in record.items), "notes": {}, "facts": [], "skip_reason": None,
                "coverage": {}}

    monkeypatch.setattr(engine, "load_context_engine", lambda: (
        build_context, lambda ep, probe, manifest: _Record(["a", "b", "c"]), _Episode, check,
        lambda *a, **k: []))
    return engine.EngineContext(withhold_untrue=withhold), built


def test_untrue_item_is_withheld_and_rechecked(monkeypatch):
    calls = []

    def check(ep, rec, ctx, others, expected_values):
        calls.append([i.id for i in rec.items])
        return ([{"item": "b", "kind": "keep_tail_unproven"}] if "b" in [i.id for i in rec.items] else []), 1

    eng, built = _engine(monkeypatch, True, check)
    ctx = eng({"id": "e1"}, [], "A", "both")
    assert built == [["a", "b", "c"], ["a", "c"]] and calls[-1] == ["a", "c"]
    assert ctx["coverage"]["withheld"] == [{"id": "b", "reason": "untrue: keep_tail_unproven"}]
    assert ctx["first_message"] == "a c"


def test_off_refuses_as_before(monkeypatch):
    eng, _ = _engine(monkeypatch, False, lambda *a, **k: ([{"item": "b", "kind": "x"}], 1))
    with pytest.raises(ValueError, match="truth: x"):
        eng({"id": "e1"}, [], "A", "both")


def test_a_violation_without_an_item_still_refuses(monkeypatch):
    eng, _ = _engine(monkeypatch, True, lambda *a, **k: ([{"item": None, "kind": "repeated_failures_count_mismatch"}], 1))
    with pytest.raises(ValueError, match="repeated_failures_count_mismatch"):
        eng({"id": "e1"}, [], "A", "both")
