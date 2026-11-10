"""Versioned local HumanEval code extraction and isolated reanalysis; no paid requests."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.evaluation.humaneval_extracted import evaluate


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--strict-run", type=Path)
    args = p.parse_args()
    if not args.validate_only and (not args.run_dir or not args.strict_run):
        p.error("Require --run-dir and --strict-run, or --validate-only")
    root = args.root.resolve()
    try:
        out = evaluate(root, (root / args.run_dir).resolve() if args.run_dir else None,
                       (root / args.strict_run).resolve() if args.strict_run else None, args.validate_only)
        print(json.dumps({"status": "complete", "output": str(out)}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped; partial reanalysis and original answers retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print(f"HumanEval reanalysis stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
