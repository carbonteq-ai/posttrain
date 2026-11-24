"""Verified, cleaned sibling solution for the plain style (docs/design.md §4.4, `solution="sibling"`).

A sibling qualifies only if it passed every native check, triggered no manifest harm (its own
assessment batches must be present to show that) and had no tool error after its last write. Its
call list is cleaned: failed calls and repeated identical reads (same tool, same arguments) are
dropped; every other read stays, because a guideline, policy or configuration read is followed
rather than copied into a later write, so its result text is no measure of whether the task
needed it; a write is dropped only when the recorded world snapshots show it was undone (every
object it touched is back to its initial state at the end) or replaced (an identical later write
exists and every object it touched was touched again later). Among the
qualified siblings the one with the fewest kept writes, then the fewest turns, is shown. A
solution that carries what a forbid line of the same context forbids (same address and content)
is withheld: the context must not contradict itself.

Verification (S1) is from the recorded native results, manifest findings and world snapshots.
The cleaned call list is then replayed in-process from the task's initial world (`replay_native`:
the fork's tool registry on the environment's world codec and deterministic simulation clock) and
every native assertion is graded on the replayed final world; a cleaned list that errors, raises or
fails a scored native check is refused (S2 for the native checks). Manifest harms are not re-assessed
on the replay: the environment's guard pipeline runs only on a sealed rollout source; the cleaning
rules keep every read and every effect the sibling's harm-free run made except calls that produced
nothing (errored, harness-rejected, exact duplicate reads) and writes the snapshots show undone or
replaced, and a failed call is not an effect or a read to any manifest source.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .record import Episode, Record
from .record.calls import READ_VERBS, episode_calls, tool_error_text
from .record.manifest import GUARD_OUTPUT, findings
from .record.model import Call, Item
from .record.observed import Corpus, atoms, normalize, value_forms
from .record.probe import _get, _objects
from .render import common


def is_read(call: Call) -> bool:
    return any(v in call.tool for v in READ_VERBS)


def _called(tool: str, args: dict) -> dict:
    """The call as the environment's tool server recorded it: the model's keyword arguments plus every
    parameter it left out whose declared default is not None (`limited_tools._call_arguments`), which is
    what the deterministic simulation seeds identifiers from."""
    import inspect

    from automationbench_v1 import limited_tools as lt

    func = lt._ZAPIER_TOOLS.get(tool)
    if func is None:
        return dict(args)
    called = dict(args)
    for name, parameter in inspect.signature(func).parameters.items():
        if name == "world" or name in called:
            continue
        if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
            called[name] = parameter.default
    return {k: v for k, v in called.items() if v is not None}


def replay_setup(sibling: Episode) -> tuple[dict | None, Any, dict[str, dict]]:
    """(the sibling's recorded initial world, the clock the environment pinned for it, the recorded call
    arguments by invocation id): replaying from the recorded world with the recorded arguments mints the
    same identifiers the sibling saw, so its later calls find the records its earlier calls created."""
    from datetime import UTC, datetime

    from .record.calls import raw_actions
    from .record.timeline import Timeline

    tl = Timeline(sibling)
    initial = tl.json(tl.initial_digest) if tl.available else None
    clock = None
    declared = ((initial or {}).get("meta") or {}).get("current_time")
    if declared:
        try:
            clock = datetime.fromisoformat(str(declared).replace("Z", "+00:00"))
            clock = clock if clock.tzinfo else clock.replace(tzinfo=UTC)
        except ValueError:
            clock = None
    return initial, clock, {inv: json.loads(a.get("arguments_json") or "null") for inv, a in raw_actions(sibling).items()
                            if isinstance(a, dict)}


def _invoke(tool: str, args: dict, world) -> Any:
    """One tool call as the environment's limited toolset made it in the recorded rollouts
    (`automationbench_v1.limited_tools.AutomationBenchLimitedToolset.invoke`): a decoded object or a
    bare number given to a `str | None` parameter goes back to the text form the tool expects, empty
    objects are not passed, and the tool runs on the world with its keyword arguments."""
    from automationbench_v1 import limited_tools as lt

    func = lt._ZAPIER_TOOLS[tool]
    kwargs = dict(args)
    json_parameters = lt._json_string_parameters(func)
    for name in lt._optional_string_parameters(func):
        value = kwargs.get(name)
        if isinstance(value, (dict, list)):
            kwargs[name] = lt._string_argument(value, name in json_parameters)
        else:
            kwargs[name] = lt._number_as_text(value)
    cleaned = {k: v for k, v in kwargs.items() if not (isinstance(v, dict) and not v)}
    return func(world=world, **cleaned)


def replay_native(task_data: dict, calls: list[Call], excluded: set[int] = frozenset(), *,
                  sibling: Episode | None = None) -> dict[str, Any]:
    """Run `calls` in-process from the task's initial world (the fork's tools on the environment's world
    codec, under its deterministic simulation clock) and grade every native assertion on the final world.
    With `sibling`, the replay starts from the sibling's recorded initial world, at the clock the
    environment pinned for it, with each call's recorded arguments (defaults filled as the tool server
    filled them), so the identifiers minted along the way are the ones the sibling's later calls name.
    {"ok": True} when every scored assertion passes; {"ok": False, "reason": ...} when a call raises or
    returns a tool error, or an assertion fails; {"ok": None, "reason": ...} when the replay cannot run
    (environment not importable, no task definition): then the recorded verification stands alone."""
    import hashlib
    from datetime import timedelta

    try:
        import automationbench.rubric  # noqa: F401  (registers the handlers)
        from automationbench import sim_runtime
        from automationbench.rubric.registry import AssertionRegistry
        from automationbench_v1.capture import canonical_json
        from automationbench_v1.simulation import SIMULATION_VERSION, simulated_setup, world_clock_base
        from automationbench_v1.world_codec import dump_world, load_world
        from automationbench_v1 import limited_tools  # noqa: F401  (the toolset that ran the recorded calls)
    except ImportError as error:
        return {"ok": None, "reason": f"environment_unavailable:{type(error).__name__}"}
    initial = task_data.get("initial_state")
    assertions = task_data.get("assertions") or []
    if not initial or not assertions:
        return {"ok": None, "reason": "no_task_definition"}

    def digest(value: Any) -> str:
        return hashlib.sha256(canonical_json(value).encode()).hexdigest()

    recorded_world, clock, recorded_arguments = replay_setup(sibling) if sibling is not None else (None, None, {})
    try:
        if recorded_world is not None:
            data = json.loads(json.dumps(recorded_world))
            clock = clock or world_clock_base(initial)
        else:
            clock = world_clock_base(initial)
            with simulated_setup(initial, clock):
                data = dump_world(load_world(json.loads(json.dumps(initial))))
    except Exception as error:  # noqa: BLE001
        return {"ok": None, "reason": f"initial_world_unavailable:{type(error).__name__}"}
    revision = 0
    for index, call in enumerate(calls):
        recorded = recorded_arguments.get(call.invocation_id) if call.invocation_id else None
        called = recorded if isinstance(recorded, dict) else _called(call.tool, call.args)
        seed = digest([SIMULATION_VERSION, digest(data), call.tool, called])
        try:
            with sim_runtime.simulated_call(seed, clock + timedelta(seconds=revision)):
                world = load_world(data)
                result = _invoke(call.tool, called, world)
                after = dump_world(world)
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "reason": f"call_raised:{call.tool}:{type(error).__name__}", "at": index, "calls_run": index}
        error = tool_error_text(result)
        if error:
            return {"ok": False, "reason": f"call_errored:{call.tool}", "at": index, "error": error[:160], "calls_run": index}
        if after != data:
            revision += 1
        data = after
    try:
        world = load_world(data)
        failed = []
        for index, assertion in enumerate(assertions):
            if index in excluded:
                continue
            verdict = AssertionRegistry.check(world, assertion)
            if verdict is not True:
                failed.append(assertion.get("type"))
    except Exception as error:  # noqa: BLE001
        return {"ok": None, "reason": f"grading_unavailable:{type(error).__name__}", "calls_run": len(calls)}
    if failed:
        return {"ok": False, "reason": "native_check_failed_on_replay", "failed": failed[:5], "calls_run": len(calls)}
    return {"ok": True, "calls_run": len(calls), "writes_changed_world": revision}


def qualify(sibling: Episode) -> tuple[bool, str | None]:
    """Whether a sibling may serve as a solution, and why not."""
    if not sibling.success:
        return False, "not_successful"
    if not sibling.native_results or not all(r.get("passed") or r.get("excluded") for r in sibling.native_results):
        return False, "native_check_failed"
    if not sibling.assessment_batches:
        return False, "no_assessment_batches"  # manifest harms cannot be ruled out
    harms = [p for p in findings(sibling.assessment_batches)[GUARD_OUTPUT]
             if p.get("kind") == "finding" and p.get("status") == "valid" and p.get("value") == 1]
    if harms:
        return False, "manifest_harm:" + ",".join(sorted({str(h.get("check_id")) for h in harms})[:3])
    calls, _, _, _ = episode_calls(sibling)
    writes = [i for i, c in enumerate(calls) if not is_read(c) and c.error is None]
    last_write = writes[-1] if writes else -1
    if any(c.error for c in calls[last_write + 1:]):
        return False, "tool_error_after_last_write"
    if not writes:
        return False, "no_write"
    return True, None


def _touched(before: dict, after: dict) -> list[tuple]:
    from automationbench_v1.capture import world_patch

    objects: list = []
    _objects(world_patch(before, after) or {}, (), before, objects)
    return [(path, kind, key) for path, kind, key in objects if kind != "removed"]


def _node(world: dict | None, path: tuple, key: Any) -> Any:
    if world is None:
        return None
    try:
        return _get(world, path + (key,))
    except (KeyError, IndexError, TypeError):
        return None


def droppable_writes(sibling: Episode, calls: list[Call]) -> set[tuple[int, int]]:
    """(turn, index) of writes the snapshots show as undone or replaced."""
    from .record.timeline import Timeline

    tl = Timeline(sibling)
    if not tl.available:
        return set()
    initial, final = tl.json(tl.initial_digest), tl.json(tl.final_digest)
    if initial is None or final is None:
        return set()
    writes = [c for c in calls if not is_read(c) and c.error is None and c.mutated]
    out: set[tuple[int, int]] = set()
    for i, w in enumerate(writes):
        before, after = tl.json(w.before_digest), tl.json(w.after_digest)
        if before is None or after is None:
            continue
        touched = _touched(before, after)
        if not touched:
            continue
        undone = all(_node(final, p, k) == _node(initial, p, k) for p, k_kind, k in [(t[0], t[1], t[2]) for t in touched])
        if undone:
            out.add((w.turn, w.index))
            continue
        later_same = [v for v in writes[i + 1:] if v.tool == w.tool and v.args == w.args]
        if not later_same:
            continue
        later_touched: set[tuple] = set()
        for v in writes[i + 1:]:
            vb, va = tl.json(v.before_digest), tl.json(v.after_digest)
            if vb is not None and va is not None:
                later_touched.update((p, k) for p, _, k in _touched(vb, va))
        if all((p, k) in later_touched for p, _, k in touched):
            out.add((w.turn, w.index))
    return out


def clean(sibling: Episode, calls: list[Call], *, drop_writes: bool = True) -> list[Call]:
    """The sibling's calls without errored calls, undone or replaced writes and exact duplicate reads
    (same tool, same arguments). Every other read is kept: whether its results were copied into a
    later write says nothing about whether the task required it (a guideline read is followed, not
    copied, and some tasks check that it happened). With `drop_writes=False` the undone or replaced
    writes stay (the replay fallback: a dropped write can change the ids later calls refer to)."""
    kept = [c for c in calls if c.error is None]
    drop = droppable_writes(sibling, kept) if drop_writes else set()
    kept = [c for c in kept if (c.turn, c.index) not in drop]
    seen: set[str] = set()
    out = []
    for c in kept:
        key = c.tool + json.dumps(c.args, sort_keys=True, default=str)
        if is_read(c) and key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


_REPLAYS: dict[tuple[str, str], dict[str, Any]] = {}  # (sibling id, cleaned calls) -> replay verdict


def _replay_cached(sibling: Episode, calls: list[Call], excluded: set[int]) -> dict[str, Any]:
    key = (sibling.id, json.dumps([[c.tool, c.args] for c in calls], sort_keys=True, default=str))
    replay = _REPLAYS.get(key)
    if replay is None:
        replay = replay_native(sibling.task_data, calls, excluded, sibling=sibling)
        if len(_REPLAYS) >= 4096:
            _REPLAYS.clear()
        _REPLAYS[key] = replay
    return replay


def candidates(record: Record, siblings: list[Episode] | None, *, include_self: bool = False) -> list[dict[str, Any]]:
    """Every sibling with its verdict; qualified ones carry their cleaned calls. The episode itself is
    a candidate only with `include_self` (a successful attempt shown its own verified call list)."""
    out = []
    for index, sibling in enumerate(siblings or []):
        if sibling.id == record.episode_id and not include_self:
            continue
        ok, reason = qualify(sibling)
        row: dict[str, Any] = {"episode_id": sibling.id, "qualified": ok, "reason": reason, "index": index}
        if ok:
            calls, _, turns, _ = episode_calls(sibling)
            kept = clean(sibling, calls)
            row.update({"calls": kept, "dropped": len([c for c in calls]) - len(kept), "turns": turns,
                        "writes": sum(not is_read(c) for c in kept)})
            # S2: the cleaned list must still pass every scored native check when replayed from the initial world
            # (one replay per sibling and cleaned list: the same sibling serves every failed attempt of its task)
            excluded = {i for i, r in enumerate(sibling.native_results) if r.get("excluded")}
            replay = _replay_cached(sibling, kept, excluded)
            if replay.get("ok") is False:
                # a dropped write changed the ids the environment minted for later calls: the list with its
                # undone or replaced writes kept (only failed calls and duplicate reads dropped) is tried
                fuller = clean(sibling, calls, drop_writes=False)
                if len(fuller) > len(kept):
                    again = _replay_cached(sibling, fuller, excluded)
                    if again.get("ok") is not False:
                        kept, replay = fuller, {**again, "fallback": "writes_kept", "cleaned_list_failed": replay.get("reason")}
                        row.update({"calls": kept, "dropped": len(calls) - len(kept), "writes": sum(not is_read(c) for c in kept)})
            row["replay"] = replay
            if replay.get("ok") is False:
                row["qualified"], row["reason"] = False, "replay_failed:" + str(replay.get("reason"))
        out.append(row)
    return out


def contradictions(calls: list[Call], items: list[Item]) -> list[str]:
    """Forbid items whose address and content a solution call carries (same app)."""
    found = []
    for item in items:
        if common.item_kind(item) != "prohibited":
            continue
        values = [v.raw for v in {**item.address, **item.content}.values()
                  if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
        if not values or not item.content:
            continue  # an address-only prohibition is checked against nothing: the task may still require the recipient
        item_app = item.effect.app.replace("_", "")
        for c in calls:
            if is_read(c):
                continue
            call_app = c.app.replace("_", "")
            if not (call_app == item_app or call_app.startswith(item_app) or item_app.startswith(call_app)):
                continue  # a CRM update naming the recipient is not the forbidden email
            text = normalize(json.dumps(c.args, sort_keys=True, default=str))
            if all(any(Corpus._form_in(f, text) for f in value_forms(v) if f) for v in values):
                found.append(item.id)
                break
    return found


HIDDEN = f"[{common.UNSEEN}]"


def hidden_values(record: Record) -> list[Any]:
    """Every scalar the attempt neither observed nor wrote, from every item's address and content
    (the values the hidden-values policy must never print); compound values contribute their hidden
    elements."""
    out: list[Any] = []
    for item in record.items:
        for value in {**item.address, **item.content}.values():
            raw = value.raw
            if isinstance(raw, dict) and "unresolved" in raw:
                continue
            if value.observed or value.written:
                continue
            if isinstance(raw, (list, dict)) and value.hidden_elements:
                hidden = set(value.hidden_elements)
                elements = ([v for i, v in enumerate(raw) if str(i) in hidden] if isinstance(raw, list)
                            else [v for k, v in raw.items() if k in hidden])
                out.extend(a for v in elements for a in atoms(v))
            else:
                out.extend(atoms(raw))
    # a number of two or more digits counts too (a grader total such as 23 must not surface in a sibling's text)
    return [v for v in out if isinstance(v, (str, int, float)) and not isinstance(v, bool)
            and (len(str(v)) >= 3 or (len(str(v)) == 2 and str(v).isdigit()))]


def _patterns(value: Any) -> list[str]:
    """Regexes for the surface forms a value may take in a call's text: the text itself with flexible
    whitespace and case; a number with or without thousands separators, decimals and a currency sign."""
    text = str(value).strip()
    # a value is masked as a whole token, never inside a word ("re[…]ed") or a longer number ("[…]000"); a
    # phrase's last word may carry an inflection ("8 registrants" for "8 registrant"), as the detector reads it
    tail = r"(?:s|es|ed|ing)?(?![A-Za-z0-9])" if " " in text else r"(?![A-Za-z0-9])"
    out = [r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(part) for part in text.split()) + tail if text else ""]
    for form in value_forms(value):
        digits = form.replace(" ", "")
        if re.fullmatch(r"\$?-?\d[\d,]*(\.\d+)?%?", digits):
            core = digits.lstrip("$-").rstrip("%")
            integer, _, fraction = core.replace(",", "").partition(".")
            groups = []
            while len(integer) > 3:
                groups.insert(0, integer[-3:])
                integer = integer[:-3]
            groups.insert(0, integer)
            number = r",?".join(re.escape(g) for g in groups)
            if fraction and int(fraction or 0) != 0:
                number += r"\." + re.escape(fraction)
            else:
                number += r"(?:\.0+)?"
            out.append(r"(?<![\d.])\$?\s?-?" + number + (r"\s?%" if digits.endswith("%") else r"(?!\.?\d)"))
    return [p for p in dict.fromkeys(out) if p]


def mask_text(text: str, values: list[Any]) -> tuple[str, int]:
    """Replace every surface form of the hidden values in `text` by the placeholder; (text, count)."""
    count = 0
    for value in sorted(values, key=lambda v: -len(str(v))):
        for pattern in _patterns(value):
            text, n = re.subn(pattern, HIDDEN, text, flags=re.IGNORECASE)
            count += n
    return text, count


def hide_values(solution: dict, record: Record, episode: Episode | None) -> tuple[dict | None, dict[str, Any]]:
    """The solution with the attempt's unobserved values masked, verified with the leak detector
    (no form of a hidden value remains); None when a value could not be masked."""
    from .checks.leak import leaks

    values = hidden_values(record)
    text, count = mask_text(solution["text"], values)
    masked = {**solution, "text": text}
    remaining = leaks(record, {"first_message": text, "notes": {}},
                      episode_messages=episode.messages if episode is not None else None,
                      task_assertions=episode.task_data.get("assertions") if episode is not None else None,
                      tool_names=tuple(episode.task_data.get("zapier_tools") or ()) if episode is not None else ())
    if remaining:
        return None, {"reason": "hidden_values_unmaskable", "masked": count, "remaining": remaining[:3]}
    return masked, {"masked": count}


def verified_solution(record: Record, siblings: list[Episode] | None, *,
                      include_self: bool = False, accept=None) -> tuple[dict | None, dict[str, Any]]:
    """(solution payload {episode_id, text, calls} or None, report)."""
    rows = candidates(record, siblings, include_self=include_self)
    report: dict[str, Any] = {"candidates": [{k: v for k, v in r.items() if k != "calls"} for r in rows],
                              "verification": "recorded_results_and_snapshots", "chosen": None, "withheld": None}
    qualified = sorted((r for r in rows if r["qualified"]), key=lambda r: (r["writes"], r["turns"], r["index"]))
    items = common.facts(record)
    fallback = None
    for row in qualified:
        clash = contradictions(row["calls"], items)
        if clash:
            report["withheld"] = {"episode_id": row["episode_id"], "reason": "contradicts_forbid_lines", "items": clash}
            continue
        text = "\n".join(f"{c.tool}({json.dumps(c.args)})" for c in row["calls"])
        if accept is not None and not accept(text):
            # the sibling's written values disagree with an expected value the feedback states: an agreeing
            # sibling is preferred; failing one, this sibling is shown with each disagreement stated
            if fallback is None:
                fallback = (row, text)
            continue
        report["chosen"] = {"episode_id": row["episode_id"], "writes": row["writes"], "turns": row["turns"],
                            "dropped_calls": row["dropped"], "replay": row.get("replay")}
        if (row.get("replay") or {}).get("ok") is True:
            report["verification"] = "recorded_results_and_snapshots+in_process_replay_of_native_checks"
        return {"episode_id": row["episode_id"], "text": text, "calls": row["calls"]}, report
    if fallback is not None:
        row, text = fallback
        report["chosen"] = {"episode_id": row["episode_id"], "writes": row["writes"], "turns": row["turns"],
                            "dropped_calls": row["dropped"], "replay": row.get("replay"), "disagrees": True}
        if (row.get("replay") or {}).get("ok") is True:
            report["verification"] = "recorded_results_and_snapshots+in_process_replay_of_native_checks"
        return {"episode_id": row["episode_id"], "text": text, "calls": row["calls"], "disagrees": True}, report
    return None, report
