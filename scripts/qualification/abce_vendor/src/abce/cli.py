"""Command line: `abce census`, `abce shapes`, and (later milestones) record / build / check."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="abce")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("census", help="counts over the design target (docs/survey.md numbers)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--refresh", action="store_true", help="reload tasks from the fork instead of the cache")
    p = sub.add_parser("shapes", help="native (type, key-set) shape table as JSON lines")
    p = sub.add_parser("adapters", help="manifest source adapter table as JSON lines")
    p = sub.add_parser("episodes", help="census of recorded episode files (excluded tasks dropped)")
    p.add_argument("--glob", default=None)
    p = sub.add_parser("context", help="teacher contexts for recorded episodes, JSON lines to stdout")
    p.add_argument("--glob", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task", action="append", default=None)
    p.add_argument("--style", default="plain", choices=["raw", "plain"])
    p.add_argument("--placement", default="both")
    p.add_argument("--expected-values", action="store_true")
    p.add_argument("--solution", default="none", choices=["none", "sibling"])
    p = sub.add_parser("record", help="build Records for recorded episodes, JSON lines to stdout")
    p.add_argument("--glob", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task", action="append", default=None)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--manifest", action="store_true")
    p.add_argument("--text", action="store_true", help="diagnostic text instead of JSON")
    p = sub.add_parser("check", help="automatic checks: coverage (738 tasks, static) | implication (episodes) | truth (sampled)")
    p.add_argument("what", choices=["coverage", "implication", "contradiction", "full", "truth"])
    p.add_argument("--per-task", type=int, default=2, help="truth: failed episodes per task per file")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--solution", default="none", choices=["none", "sibling"])
    p.add_argument("--hidden-values", action="store_true", help="truth: render with expected values hidden and run the leak test")
    p.add_argument("--out", default=None, help="truth: write the full report (rows) to this JSON file")
    p.add_argument("--placement", default="decision_point")
    p.add_argument("--expected-values", action="store_true")
    p.add_argument("--glob", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shapes", action="store_true", help="coverage: also print the per-shape table")
    p.add_argument("--out-dir", default=".cache/full_pass")
    p.add_argument("--sample", type=int, default=3, help="full: failed episodes per task per file (0 = all)")
    args = parser.parse_args(argv)
    if getattr(args, "sample", None) == 0:
        args.sample = None

    from . import survey, tasks

    if args.command == "census":
        if args.refresh:
            tasks.load_tasks.cache_clear()
            tasks.load_tasks(refresh=True)
        c = survey.census()
        print(json.dumps(c, indent=1, sort_keys=True) if args.json else survey.format_census(c))
    elif args.command == "shapes":
        for row in survey.shape_table():
            print(json.dumps(row))
    elif args.command == "adapters":
        for row in survey.adapter_table():
            print(json.dumps(row))
    elif args.command == "episodes":
        from .record import episode_census, episode_files

        c = episode_census(episode_files(args.glob))
        c["episodes_per_task"] = len(c["episodes_per_task"])
        print(json.dumps(c, indent=1, sort_keys=True))
    elif args.command == "context":
        from .context import build_context
        from .record import build_record, episode_files, iter_episodes

        episodes = list(iter_episodes(episode_files(args.glob), tasks=set(args.task) if args.task else None, limit=args.limit))
        by_task: dict = {}
        for ep in episodes:
            by_task.setdefault(ep.task, []).append(ep)
        for ep in episodes:
            record = build_record(ep, probe=True, manifest=True)
            print(json.dumps(build_context(record, by_task[ep.task], style=args.style, placement=args.placement,
                                           expected_values=args.expected_values, episode=ep, solution=args.solution)))
    elif args.command == "record":
        from .record import build_record, episode_files, iter_episodes

        for ep in iter_episodes(episode_files(args.glob), tasks=set(args.task) if args.task else None, limit=args.limit):
            record = build_record(ep, probe=args.probe, manifest=args.manifest)
            if args.text:
                from .render.debug import record_text

                print(record_text(record))
            else:
                print(record.model_dump_json())
    elif args.command == "check":
        if args.what == "coverage":
            from .coverage import static_coverage

            report = static_coverage()
            print(json.dumps(report["summary"], indent=1, sort_keys=True))
            if args.shapes:
                for row in report["shapes"]:
                    print(json.dumps(row))
        elif args.what == "full":
            from .checks.full_pass import full_pass
            from .record import episode_files

            print(json.dumps(full_pass(episode_files(args.glob), out_dir=args.out_dir, sample=args.sample,
                                       placement=args.placement, expected_values=args.expected_values), indent=1))
        elif args.what == "truth":
            from .checks.truth import truth_pass

            report = truth_pass(per_task=args.per_task, seed=args.seed, solution=args.solution,
                                expected_values=not args.hidden_values, out_path=args.out)
            print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
            for row in report["rows"][:40]:
                print(json.dumps(row))
        elif args.what == "contradiction":
            from .checks.contradiction import contradiction_report
            from .record import episode_files

            report = contradiction_report(episode_files(args.glob), limit=args.limit)
            print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
            for row in report["rows"][:50]:
                print(json.dumps(row))
        else:
            from .checks.implication import implication_report
            from .record import episode_files

            report = implication_report(episode_files(args.glob), limit=args.limit)
            print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
