"""MATH500 Jev: offline prepare, explicit paid selection, then offline selection audit."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import math500_jev as workflow
from jev_dllm.judges.credentials import load_api_key


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "live", "audit"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--approved-calls", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = lambda p: (root / p).resolve() if p else None
    try:
        if args.action == "prepare":
            if not args.run_dir:
                parser.error("prepare requires --run-dir")
            out = workflow.prepare(root, path(args.run_dir), path(args.plan_dir))
            plan, _, _, _ = workflow.inspect(root, out)
            print(json.dumps({"planned_paid_calls": plan["question_count"], "candidate_count": plan["candidate_count"],
                              "paid_calls_sent": 0, "plan_directory": str(out)}, indent=2))
        elif args.action == "check":
            if not args.plan_dir:
                parser.error("check requires --plan-dir")
            plan, _, _, _ = workflow.inspect(root, path(args.plan_dir))
            print(json.dumps({"status": "passed", "planned_paid_calls": plan["question_count"],
                              "candidate_count": plan["candidate_count"], "paid_calls_sent": 0}, indent=2))
            return 0
        elif args.action == "live":
            if not args.plan_dir or args.approved_calls is None:
                parser.error("live requires --plan-dir and --approved-calls")
            # Validate before reading credentials, and before any network request.
            workflow.inspect(root, path(args.plan_dir))
            load_api_key(path(args.env_file))
            if not os.environ.get("JEV_API_KEY"):
                raise ValueError("Set JEV_API_KEY in the ignored .env file")
            out = workflow.execute(root, path(args.plan_dir), args.approved_calls)
        else:
            if not args.plan_dir:
                parser.error("audit requires --plan-dir")
            out = workflow.audit_live(root, path(args.plan_dir))
        print(json.dumps({"status": "complete", "action": args.action, "output": str(out)}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped; saved evidence retained. Do not restart a paid batch blindly.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"MATH500 Jev stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
