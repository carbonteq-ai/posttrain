"""The runner interface: `build_context(record, siblings, *, style, placement, expected_values)`.

Returns the training runner's context dict — `first_message` (string) and `notes` (turn number ->
text appended to the tool/user message before that assistant turn) — plus `skip_reason` when an
episode gets no context, `facts` (item ids), `bad_spans` and `solution_episode`.
"""

from __future__ import annotations

import json
from typing import Any

from .record import Episode, Record, build_record
from .render import common, plain, raw


SOLUTION_PARTIAL_HEADING = "Solution that passed the task's checks:"


def solution_from(sibling: Episode | None, style: str) -> dict | None:
    """A successful sibling as the 'Correct solution' payload (raw: the handover's JSON of its
    assistant and tool messages; plain: its tool calls, one per line)."""
    if sibling is None or not sibling.success:
        return None
    if style == "raw":
        text = json.dumps([m for m in sibling.messages if m.get("role") in ("assistant", "tool")], sort_keys=True)
    else:
        lines = []
        for message in sibling.messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = call.get("name") or function.get("name")
                args = call.get("arguments", function.get("arguments"))
                lines.append(f"{name}({args if isinstance(args, str) else json.dumps(args)})")
        text = "\n".join(lines)
    return {"episode_id": sibling.id, "text": text}


def build_context(record: Record, siblings: list[Episode] | None = None, *, style: str = "plain",
                  placement: str = "both", expected_values: bool = False, episode: Episode | None = None,
                  solution: str = "none", include_manifest: bool = True, fix_hints: Any = False,
                  line_budget: int | None = plain.LINE_BUDGET, char_budget: int | None = plain.CHAR_BUDGET,
                  order: str = plain.DEFAULT_ORDER, not_done_source: str = plain.DEFAULT_NOT_DONE_SOURCE,
                  final_note: bool = False) -> dict[str, Any]:
    """The teacher context of one recorded attempt.

    Plain-style settings (Fix 3; the defaults are the released behaviour, byte for byte):
    `line_budget` — at most this many statement lines in the first message (and per note), default 8;
    `char_budget` — at most this many characters of statement lines in the first message, default 2,400;
    either None means no limit (nothing is dropped, no "Not listed" line);
    `order` — "worst_first" (default: harms, unrecovered tool errors, missing work, recovered tool errors) or
    "turn" (the same lines kept, shown by the first turn they name). The raw style ignores them.

    Correction r0 (plain style only; the defaults change nothing, byte for byte):
    `not_done_source` — "all" (default) or "benchmark": the "Not done" line of a failed requirement whose only source is
    the task's rule file (an unmerged manifest item) is withheld, with every note and hint derived from it; lines backed
    by a benchmark assertion (`plain.benchmark_backed`), harms and tool errors are unchanged;
    `final_note` — False (default) or True: a note of kind "final_not_done" at the turn of the attempt's final reply
    (`notes[turn]`, placed like any other note; described in `context["final_note"]`) listing the benchmark-backed
    "Not done" items, "Still not done at this point: <item>; <item>."; none on a successful attempt, when nothing
    required is missing or when the attempt has no final reply; not bound by the line budget. Needs placement "notes"
    or "both".

    `fix_hints` (plain style with expected values shown) selects the version-(c) line kinds by name (abce.fix
    FIX_HINT_KINDS): "value_location", "wrong_value", "missed_write", "unrequested_change", "call_fix" (a failed call's
    line followed by the proven fix), "content_at_call" (opt-in: at the call that wrote the addressed object, the
    required content it lacks), "why_clause" (N1 clauses), "solution_note". True = all of them except the opt-in
    "content_at_call" (the earlier boolean's whole family, byte for byte); False, None or an empty collection = none (no version (c)). Any selection
    turns on version (c): its statement forms and repairs, the solution heading of R7 and the checker's stricter rows."""
    if style not in common.STYLES or placement not in common.PLACEMENTS:
        raise ValueError(f"unknown style {style!r} or placement {placement!r}")
    if not_done_source not in plain.NOT_DONE_SOURCES:
        raise ValueError(f"not_done_source must be one of {plain.NOT_DONE_SOURCES}, not {not_done_source!r}")
    if not isinstance(final_note, bool):
        raise ValueError(f"final_note must be True or False, not {final_note!r}")
    if (not_done_source != plain.DEFAULT_NOT_DONE_SOURCE or final_note) and style != "plain":
        raise ValueError("not_done_source and final_note are settings of the plain style")
    if final_note and placement not in ("notes", "both"):
        raise ValueError(f"final_note needs per-turn notes (placement 'notes' or 'both'), not {placement!r}")
    from .fix import selected_kinds

    kinds = selected_kinds(fix_hints)
    fix_hints = kinds is not None
    if fix_hints and (style != "plain" or not expected_values):
        raise ValueError("fix_hints needs style='plain' and expected_values=True")
    if solution not in ("none", "sibling"):
        raise ValueError("solution must be 'none' or 'sibling'")
    for name, limit in (("line_budget", line_budget), ("char_budget", char_budget)):
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError(f"{name} must be a positive integer or None (no limit), not {limit!r}")
    if order not in plain.ORDERS:
        raise ValueError(f"order must be one of {plain.ORDERS}, not {order!r}")
    messages = episode.messages if episode is not None else None
    chosen = None
    solution_report = None
    if solution == "sibling":
        if style == "raw":
            chosen = solution_from(next((s for s in siblings or [] if s.success and s.id != record.episode_id), None), style)
        else:
            from .solution import hide_values, verified_solution

            from .fix import solution_agrees

            chosen, solution_report = verified_solution(record, siblings, accept=lambda text: not solution_agrees(text, record))
            if chosen is not None and fix_hints:
                # R7: the heading claims only what the solution's own findings prove
                from .fix import sibling_manifest_outcomes

                sibling_ep = next((s for s in siblings or [] if s.id == chosen["episode_id"]), None)
                if any(v == 0 for v in sibling_manifest_outcomes(sibling_ep).values()):
                    chosen = {**chosen, "heading": SOLUTION_PARTIAL_HEADING}
            if chosen is not None and chosen.get("disagrees"):
                from .fix import disagreement_notes, sibling_outcomes

                sibling_ep = next((s for s in siblings or [] if s.id == chosen["episode_id"]), None)
                from .fix import sibling_manifest_outcomes

                notes_, unstated = disagreement_notes(chosen["calls"], record, sibling_outcomes(sibling_ep),
                                                      sibling_manifest_outcomes(sibling_ep)
                                                      if fix_hints and "solution_note" in kinds else None)
                if True:  # fix round: solution notes removed; a solution that disagrees is withheld
                    solution_report["withheld"] = {"episode_id": chosen["episode_id"],
                                                   "reason": "disagreement_cannot_be_stated" if unstated else "disagrees_with_expected_values"}
                    solution_report["chosen"] = None
                    chosen = None
                else:
                    chosen["notes"] = notes_
                    solution_report["disagreement_notes"] = len(notes_)
            if chosen is not None and not expected_values:
                # hidden-values policy: the sibling's arguments may carry values the attempt never saw
                chosen, hidden = hide_values(chosen, record, episode)
                solution_report["hidden_values"] = hidden
                if chosen is None:
                    solution_report["withheld"] = {"episode_id": solution_report["chosen"]["episode_id"],
                                                   "reason": hidden["reason"]}
                    solution_report["chosen"] = None
    if style == "plain_matched":
        raise ValueError("plain_matched exists only for feedback_for_action")
    if style == "raw":
        context = raw.render(record, placement=placement, expected_values=expected_values, episode_messages=messages,
                             solution=chosen, include_manifest=include_manifest)
    else:
        hint_data = None
        if fix_hints:
            from .fix import hints as fix_hints_for
            from .record.build import task_contract
            from .record.calls import episode_calls
            from .record.timeline import Timeline

            sibling_calls = [episode_calls(s)[0] for s in (siblings or []) if s.id != record.episode_id]
            hint_data = {"initial_state": episode.task_data.get("initial_state") if episode is not None else None,
                         "timeline": Timeline(episode) if episode is not None else None, "sibling_calls": sibling_calls,
                         "contract": task_contract(episode) if episode is not None else None, "build": fix_hints_for,
                         "siblings": [s for s in (siblings or []) if s.id != record.episode_id], "solution": chosen,
                         "assertions": list(episode.task_data.get("assertions") or []) if episode is not None else [],
                         "request_text": common.request_text(record, messages), "kinds": kinds}
        context = plain.render(record, placement=placement, expected_values=expected_values, episode_messages=messages,
                               solution=chosen, fix_hints=hint_data, budget=line_budget, chars=char_budget, order=order,
                               not_done_source=not_done_source, final_note=final_note)
    if solution_report is not None:
        context["coverage"]["solution"] = solution_report
    return context


def successful_reference(record: Record, siblings: list[Episode] | None, episode: Episode | None) -> Episode | None:
    """The handover's rule: the episode itself when it succeeded, else the first successful sibling."""
    if record.success and episode is not None:
        return episode
    for sibling in siblings or []:
        if sibling.success and sibling.id == record.episode_id:
            return sibling
    return next((s for s in siblings or [] if s.success), None)


def feedback_for_action(record: Record, siblings: list[Episode] | None, action_index: int, *, style: str = "raw",
                        expected_values: bool = True, episode: Episode | None = None,
                        include_manifest: bool = False) -> dict[str, Any]:
    """The teacher text placed as a separate user message right before assistant action `action_index`
    (0-based ordinal of the sampled assistant turns), in the handover's scheme.

    Returns {"text": str | None, "skip_reason": str | None, "turn": int, "solution_episode": str | None}.
    raw: byte-identical to the handover's `feedback_text` (with expected values shown and no manifest
    rows); plain: grammar lines for the same facts; the 'Correct solution' rule is the same for both."""
    if style not in common.STYLES:
        raise ValueError(f"unknown style {style!r}")
    turn = action_index + 1
    if action_index < 0 or turn > record.turns:
        return {"text": None, "skip_reason": "no_such_action", "turn": turn, "solution_episode": None}
    reference = successful_reference(record, siblings, episode)
    if style == "plain_matched":
        # Exactly the raw text's information (failed native checks, the action's tool responses, the
        # same reference rollout and closing), presented as plain lines; values always shown.
        text = plain.feedback_for_action_matched(
            record, turn, solution_messages=reference.messages if reference is not None else None,
            observations=raw.action_observations(record, turn))
        return {"text": text, "skip_reason": None, "turn": turn,
                "solution_episode": reference.id if reference is not None else None}
    if style == "raw":
        solution = solution_from(reference, style) if reference is not None else None
        text = raw.feedback_for_action(record, turn, expected_values=expected_values, solution=solution,
                                       include_manifest=include_manifest)
    else:
        # plain: only a verified, cleaned sibling is shown as the solution (abce.solution); the episode
        # itself, when it succeeded, is shown as its own verified call list
        from .solution import hide_values, verified_solution

        pool = list(siblings or [])
        if record.success and episode is not None and all(s.id != episode.id for s in pool):
            pool.append(episode)
        solution, _report = verified_solution(record, pool, include_self=record.success)
        if solution is not None and not expected_values:
            solution, _hidden = hide_values(solution, record, episode)
        text = plain.feedback_for_action(record, turn, expected_values=expected_values, solution=solution)
    reason = None if text else "nothing_to_say"
    return {"text": text, "skip_reason": reason, "turn": turn, "solution_episode": solution["episode_id"] if solution else None}


def context_for_episode(episode: Episode, siblings: list[Episode] | None = None, **switches) -> dict[str, Any]:
    """Convenience: Record + context in one call (probing and manifests on)."""
    record = build_record(episode, siblings, probe=True, manifest=True)
    return build_context(record, siblings, episode=episode, **switches)
