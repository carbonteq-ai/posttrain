"""Teacher feedback from the frozen context engine (D42/D43 style test), for `sdpo_ab_handover.handover_rows`.

The engine is vendored at `abce_vendor/` (source and its two name lists, copied from the separate
automationbench-context-engine repository at the commit in `abce_vendor/VERSION`), so the GPU job runs exactly the
frozen version. `ABCE_SRC` points at another checkout instead (local tests only).

Every row is checked: the engine's raw style must equal the handover's `feedback_text` for the same action, which
proves the engine read the episode the way the handover recipe does in this environment. A mismatch, or an action
the engine cannot describe, stops the run: the style test changes only the wording of the same facts.
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "abce_vendor"
STYLES = ("raw", "plain_matched")
# Correction r0: settings of the corrected engine's `build_context` that the runner may pass (EngineContext `options`)
ENGINE_OPTIONS = ("line_budget", "char_budget", "not_done_source", "final_note")
NOT_DONE_SOURCES = ("all", "benchmark")


def check_engine_options(options):
    """The given engine options, checked (unknown names and impossible values refused before any engine call)."""
    unknown = sorted(set(options) - set(ENGINE_OPTIONS))
    if unknown:
        raise ValueError(f"unknown engine options {unknown}")
    for name in ("line_budget", "char_budget"):
        if name in options:
            value = options[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None (no limit), not {value!r}")
    if "not_done_source" in options and options["not_done_source"] not in NOT_DONE_SOURCES:
        raise ValueError(f"not_done_source must be one of {NOT_DONE_SOURCES}, not {options['not_done_source']!r}")
    if "final_note" in options and not isinstance(options["final_note"], bool):
        raise ValueError(f"final_note must be True or False, not {options['final_note']!r}")
    return dict(options)


def missing_manifests(task_names):
    """The sampled tasks whose manifest (task contract) the engine cannot load. `load_engine` points the engine at
    the environment; when that path is wrong every contract is silently None (abce.record.build.task_contract), so a
    job that needs them checks here and stops."""
    load_engine()
    from abce import paths
    from abce.tasks import get_task

    missing = []
    for name in task_names:
        try:
            contract = get_task(name).contract()
        except (KeyError, FileNotFoundError) as error:
            missing.append(f"{name} ({type(error).__name__})")
            continue
        if not contract:
            missing.append(f"{name} (empty)")
    if missing and not Path(str(getattr(paths, "ENVIRONMENT", ""))).is_dir():
        missing.append(f"environment path {getattr(paths, 'ENVIRONMENT', None)} absent")
    return missing


def engine_supports(names):
    """Whether the loaded engine's `build_context` takes every setting in `names` (the vendored engine before
    correction r0 takes none of ENGINE_OPTIONS)."""
    import inspect

    build_context = load_context_engine()[0]
    return set(names) <= set(inspect.signature(build_context).parameters)


def engine_source():
    return Path(os.environ["ABCE_SRC"]) if os.environ.get("ABCE_SRC") else VENDOR / "src"


def engine_version():
    version = VENDOR / "VERSION"
    return version.read_text().strip() if version.exists() and not os.environ.get("ABCE_SRC") else "unvendored"


def load_engine():
    source = engine_source()
    if not (source / "abce").is_dir():
        raise FileNotFoundError(f"context engine not found at {source}")
    if os.environ.get("ABCE_SRC") is None:
        os.environ.setdefault("ABCE_CACHE", "/tmp/abce-cache")
        # the GPU container ships the environment at /env; on the laptop the research checkout sits next to this
        # repository. The vendored copy cannot find either by itself, and without it every manifest is silently
        # missing (abce.record.build.task_contract), so the path is set explicitly.
        for environment in (Path("/env/environments/automationbench_v1"),
                            Path(__file__).resolve().parents[3] / "verifiers-environments/environments/automationbench_v1"):
            if environment.is_dir():
                os.environ.setdefault("ABCE_ENVIRONMENT", str(environment))
                break
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from abce.context import feedback_for_action
    from abce.record import build_record
    from abce.record.episode import Episode

    return feedback_for_action, build_record, Episode


def load_context_engine():
    """(build_context, build_record, Episode, truth check_episode, leak finder) from the engine."""
    load_engine()
    from abce.checks.leak import leaks
    from abce.checks.truth import check_episode
    from abce.context import build_context
    from abce.record import build_record
    from abce.record.episode import Episode

    return build_context, build_record, Episode, check_episode, leaks


class EngineContext:
    """`build_context` replacement for `train_arm` (D44 per-tool-call scheme): the engine's plain context for one
    episode (first message + per-turn notes, expected values hidden; arm B adds the engine's verified sibling
    solution). Every context is checked by the engine's truth test (and, with values hidden, its leak test) before
    use; a context that fails raises ValueError, which `train_arm` records as an exclusion (no stop). Returns None when the engine has nothing to say.
    Signature matches `sdpo_ab_context.build_context(record, siblings, arm, placement, contract, variant=...)`.
    Recipe V: `expected_values=True` shows the expected values (the truth check still runs on every context; the
    leak test is skipped, since showing the values is the point); `fix_hints` is a list of the engine's fix-hint
    kind names (abce.fix.FIX_HINT_KINDS; the study approved only "call_fix") passed to the engine as is, and needs
    values shown; True (all kinds) is refused, so every kind is named. The defaults (hidden, no hints) are recipe U's
    behaviour.
    Correction r0: `options` are further `build_context` settings of the corrected engine (ENGINE_OPTIONS:
    line_budget and char_budget, a positive int or None for no limit; not_done_source "all" | "benchmark";
    final_note True | False), passed only when given, so a call without them is exactly the earlier one."""

    def __init__(self, expected_values=False, pool=False, fix_hints=False, options=None):
        if fix_hints is True or isinstance(fix_hints, str):
            raise ValueError(f"fix_hints must be a list of kind names, not {fix_hints!r}")
        fix_hints = list(fix_hints or [])
        if fix_hints and not expected_values:
            raise ValueError("fix hints need the expected values shown (--engine-values shown)")
        self.expected_values = expected_values
        self.fix_hints = fix_hints
        # passed only when on, so the default call to the engine is exactly recipe U's
        self.hint_option = {"fix_hints": list(fix_hints)} if fix_hints else {}
        self.options = check_engine_options(options or {})
        self.hint_option = {**self.hint_option, **self.options}
        self.build_context, self.build_record, self.Episode, self.check, self.leaks = load_context_engine()
        if fix_hints:
            load_engine()
            from abce.fix import selected_kinds

            selected_kinds(fix_hints)  # unknown kind names refused before any engine call
        self.stats = Counter()
        # D47 recipe P: successful attempts of earlier batches (and given folders), per task. A failed attempt whose
        # own batch offers no verified solution is offered these as further siblings (arm B only).
        self.pool = defaultdict(list) if pool else None

    def add_to_pool(self, records):
        if self.pool is None:
            return
        for record in records:
            episode = self.Episode(record=record)
            if episode.success and all(r["id"] != record["id"] for r in self.pool[episode.task]):
                self.pool[episode.task].append(record)
                self.stats["pool_records"] += 1

    def __call__(self, record, siblings, arm, placement, contract=None, variant=None):
        if placement != "both":
            raise ValueError(f"engine contexts are built with placement both, not {placement!r}")
        episode = self.Episode(record=record)
        others = [self.Episode(record=r) for r in siblings if r["id"] != record["id"]]
        engine_record = self.build_record(episode, probe=True, manifest=True)
        self.stats["records"] += 1
        self.stats["manifest_items"] += sum(getattr(i, "origin", None) == "manifest" for i in engine_record.items)
        for note in engine_record.notes:
            self.stats["record_note:" + " ".join(str(note).split()[:4])] += 1
        context = self.build_context(engine_record, others, style="plain", placement="both",
                                     expected_values=self.expected_values, episode=episode,
                                     solution="sibling" if arm == "B" else "none", **self.hint_option)
        if self.pool is not None and arm == "B" and not episode.success and not context.get("solution_episode"):
            known = {r["id"] for r in siblings} | {record["id"]}
            extra = [self.Episode(record=r) for r in self.pool.get(episode.task, []) if r["id"] not in known]
            if extra:
                pooled = self.build_context(engine_record, others + extra, style="plain", placement="both",
                                            expected_values=self.expected_values, episode=episode, solution="sibling",
                                            **self.hint_option)
                if pooled.get("solution_episode"):
                    context, others = pooled, others + extra
                    self.stats["solution_from_pool"] += 1
        for entry in (context.get("coverage") or {}).get("suppressed") or []:
            self.stats["suppressed:" + str(entry.get("reason"))] += 1
        if context.get("skip_reason"):
            self.stats["skipped:" + context["skip_reason"]] += 1
            return None
        violations, checked = self.check(episode, engine_record, context, others, expected_values=self.expected_values)
        self.stats["lines_checked"] += checked
        if violations:
            self.stats["excluded_truth"] += 1
            raise ValueError("truth: " + "; ".join(sorted({v["kind"] for v in violations})))
        found = [] if self.expected_values else self.leaks(engine_record, context, episode_messages=episode.messages)
        if found:
            self.stats["excluded_leak"] += 1
            raise ValueError("leak: " + "; ".join(sorted({str(f.get("key")) for f in found})))
        self.stats["contexts"] += 1
        self.stats["with_solution"] += bool(context.get("solution_episode"))
        context["episode_id"] = record["id"]
        return context


class EngineFeedback:
    """`feedback` hook for `handover_rows`: the engine's text in `style` for one action, after checking that the
    engine's raw text equals the handover's. Call `reset()` after each update (records are per update)."""

    def __init__(self, style):
        if style not in STYLES:
            raise ValueError(f"unknown engine style {style!r}")
        self.style = style
        self.feedback_for_action, self.build_record, self.Episode = load_engine()
        self.records = {}
        self.stats = Counter()

    def reset(self):
        self.records, self.stats = {}, Counter()

    def __call__(self, record, task_records, ordinal, default):
        episode = self.Episode(record=record)
        if episode.id not in self.records:
            self.records[episode.id] = self.build_record(episode, probe=True, manifest=False)
        engine_record = self.records[episode.id]
        siblings = [self.Episode(record=r) for r in task_records]
        raw = self.feedback_for_action(engine_record, siblings, ordinal, style="raw", expected_values=True,
                                       episode=episode)
        if raw["text"] != default:
            raise ValueError(f"engine raw text differs from the handover's: episode {episode.id}, action {ordinal}")
        self.stats["raw_equal"] += 1
        if self.style == "raw":
            return raw["text"]
        out = self.feedback_for_action(engine_record, siblings, ordinal, style=self.style, expected_values=True,
                                       episode=episode)
        if not out["text"]:
            raise ValueError(f"engine gave no {self.style} text where the handover has one: episode {episode.id}, "
                             f"action {ordinal} ({out['skip_reason']})")
        self.stats[self.style] += 1
        return out["text"]


class EngineTeacher:
    """Correction b, `--style engine` for the handover phase: the final engine's full context for one episode, in
    place of the handover's raw native-check text, while every other part of the handover scheme stays `--style raw`'s.

    The context is recipe V's (`EngineContext`): plain style, placement both, expected values shown (the handover's
    raw text shows them too), no fix hints (the handover gives none), the engine's own verified same-batch sibling as
    the solution (arm B: the handover's reference is also a same-batch success), and no cross-batch solution pool (the
    handover has none, and it would add information the raw teacher never had). The engine's truth check runs on
    every context and cannot be switched off; a context that fails it is not used.

    `context(record, siblings)` returns (context, None) or (None, reason); the runner then keeps the raw teacher for
    that row and counts the reason (no context, truth, builder error). One instance per run; `reset()` per update."""

    def __init__(self, fix_hints=False, options=None):
        # correction r0: --engine-fix-hints and the corrected engine's options reach the handover's engine teacher
        # too; without them the call is exactly correction b's
        self.engine = EngineContext(expected_values=True, pool=False, fix_hints=fix_hints, options=options)
        self.cache = {}
        self.stats = Counter()

    def reset(self):
        self.cache, self.stats, self.engine.stats = {}, Counter(), Counter()

    def context(self, record, siblings):
        if record["id"] not in self.cache:
            try:
                built = self.engine(record, siblings, "B", "both")
                self.cache[record["id"]] = (built, None) if built is not None else (None, "no_context")
            except ValueError as error:  # the engine's truth check (or, values hidden, its leak test) failed
                self.cache[record["id"]] = (None, "untrue: " + str(error))
            except Exception as error:  # noqa: BLE001 - D45: an engine crash never stops training
                self.engine.stats["builder_error"] += 1
                self.cache[record["id"]] = (None, f"builder_error: {type(error).__name__}: {error}")
        return self.cache[record["id"]]
