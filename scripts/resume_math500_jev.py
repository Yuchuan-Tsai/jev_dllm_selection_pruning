"""Inspect or explicitly execute a further MATH500 Jev continuation."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import math500_jev_resume as workflow
from jev_dllm.judges.credentials import load_api_key


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "live", "audit"))
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approved-calls", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = lambda p: (root / p).resolve() if p else None
    try:
        if args.action == "prepare":
            if not args.source_plan:
                parser.error("prepare requires --source-plan")
            out = workflow.prepare(root, path(args.source_plan))
        else:
            if not args.plan:
                parser.error("Require --plan")
            plan = path(args.plan)
            amendment, _, _ = workflow.inspect(root, plan)
            if args.action == "check":
                print(json.dumps(amendment, indent=2))
                return 0
            if args.action == "live":
                if args.approved_calls != amendment['additional_paid_calls']:
                    parser.error("--approved-calls must equal the exact planned additional count")
                load_api_key(path(args.env_file))
                if not os.environ.get("TYPESAFE_API_KEY"):
                    raise ValueError("Set TYPESAFE_API_KEY in the ignored .env file")
                out = workflow.execute(root, plan, args.approved_calls)
            else:
                out = workflow.audit_live(root, plan)
        print(json.dumps({"status": "complete", "action": args.action, "output": str(out)}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped; original and partial continuation evidence retained. Inspect before any further request.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"MATH500 resume stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
