"""Local, isolated HumanEval scoring. No GPU model or paid API work."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.evaluation.humaneval_isolated import evaluate


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--run-dir", type=Path)
    args = p.parse_args()
    if not args.validate_only and not args.run_dir:
        p.error("Provide --run-dir or --validate-only")
    root = args.root.resolve()
    try:
        path = evaluate(root, (root / args.run_dir).resolve() if args.run_dir else None, args.validate_only)
        print(json.dumps({"status": "complete", "output": str(path)}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; partial scoring evidence retained. No source answers were changed.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print(f"HumanEval scoring stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
