#!/usr/bin/env python3
"""Prepare, check, run and analyze Week 4 partial-draft pruning replays."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import week4_pruning


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="action", required=True)
    cmd = commands.add_parser("prepare")
    cmd.add_argument("--task", choices=("gsm8k", "math"), default="gsm8k")
    cmd.add_argument("--pool-dir", type=Path)
    cmd.add_argument("--first", type=int, default=20)
    cmd.add_argument("--checkpoint", type=int, default=2)
    cmd.add_argument("--keep", type=int, default=8)
    cmd.add_argument("--allow-tokenizer-download", action="store_true")
    commands.add_parser("check").add_argument("--plan-dir", type=Path, required=True)
    cmd = commands.add_parser("live")
    cmd.add_argument("--plan-dir", type=Path, required=True)
    cmd.add_argument("--approved-calls", type=int, required=True)
    cmd.add_argument("--env-file", type=Path, default=Path(".env"))
    commands.add_parser("analyze").add_argument("--plan-dir", type=Path, required=True)
    cmd = commands.add_parser("compare")
    cmd.add_argument("--analysis-dirs", type=Path, nargs=3, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "prepare":
            pool_dir = args.pool_dir or Path(week4_pruning.DEFAULT_POOLS[args.task])
            out = week4_pruning.prepare(root, args.task, (root / pool_dir).resolve(), args.first,
                args.checkpoint, args.keep, args.allow_tokenizer_download)
            plan = json.loads((out / "plan.json").read_text())
            result = {"status": "prepared", "plan_directory": str(out),
                      "planned_calls": plan["planned_calls"], "paid_calls_sent": 0}
        elif args.action == "check":
            plan, calls = week4_pruning.check(root, (root / args.plan_dir).resolve())
            result = {"status": "passed", "task": plan["task"], "planned_calls": len(calls),
                      "checkpoint": plan["checkpoint_completed_blocks"], "keep": plan["keep"],
                      "paid_calls_sent": 0}
        elif args.action == "live":
            out = week4_pruning.live(root, (root / args.plan_dir).resolve(), args.approved_calls,
                                     (root / args.env_file).resolve())
            result = {"status": "complete", "live_directory": str(out)}
        else:
            if args.action == "analyze":
                out = week4_pruning.analyze(root, (root / args.plan_dir).resolve())
                result = {"status": "complete", "analysis_directory": str(out),
                          "metrics": json.loads((out / "metrics.json").read_text())}
            else:
                out = week4_pruning.compare(root, [(root / d).resolve() for d in args.analysis_dirs])
                result = {"status": "complete", "comparison_directory": str(out),
                          "comparison": json.loads((out / "comparison.json").read_text())}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
