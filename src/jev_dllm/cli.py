import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
from .config import load
from .data.splits import validate_splits
from .experiments.smoke import run
from .schema import Question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jev × dLLM research scaffold (no remote calls)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local runtime; do not install anything")
    check = commands.add_parser("check-config", help="Validate config and list unresolved fields")
    check.add_argument("--config", type=Path)
    check.add_argument("--require-resolved", action="store_true")
    smoke = commands.add_parser("smoke", help="Run synthetic offline plumbing checks, not a benchmark")
    smoke.add_argument("--output", type=Path)
    splits = commands.add_parser("check-splits", help="Validate public dev/heldout JSON arrays")
    splits.add_argument("--dev", type=Path, required=True)
    splits.add_argument("--heldout", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                "machine": platform.machine(), "local_scaffold_supported": sys.version_info >= (3, 11),
                "optional_packages_installed": {m: importlib.util.find_spec(m) is not None
                    for m in ("torch", "transformers", "datasets")},
                "real_generator_adapter": "pending", "jev_transport": "pending",
                "note": "Package presence does not establish CUDA/BF16/model readiness."}, indent=2))
        elif args.command == "check-config":
            config, pending = load(args.config or args.root / "configs/week01/protocol.json")
            print(json.dumps({"structurally_valid": True, "frozen": config["frozen"],
                              "unresolved_fields": pending, "runtime_readiness_verified": False}, indent=2))
            return int(args.require_resolved and bool(pending))
        elif args.command == "check-splits":
            dev = [Question(**row) for row in json.loads(args.dev.read_text())]
            heldout = [Question(**row) for row in json.loads(args.heldout.read_text())]
            print(json.dumps(validate_splits(dev, heldout), indent=2))
        else:
            path = run(args.root, args.output)
            print(f"Synthetic smoke check complete (NOT a benchmark result): {path}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
