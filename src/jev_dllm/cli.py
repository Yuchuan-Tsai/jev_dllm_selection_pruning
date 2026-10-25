import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import os
import hashlib
from .config import load
from .data.splits import validate_splits
from .experiments.smoke import run
from .schema import Question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jev × dLLM selection study; model and live API commands are explicit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local runtime; do not install anything")
    commands.add_parser("gsm8k500-check", help="Verify the scoped frozen protocol without GPU or API work")
    check = commands.add_parser("check-config", help="Validate config and list unresolved fields")
    check.add_argument("--config", type=Path)
    check.add_argument("--require-resolved", action="store_true")
    smoke = commands.add_parser("smoke", help="Run synthetic offline plumbing checks, not a benchmark")
    smoke.add_argument("--output", type=Path)
    smoke.add_argument("--keep-artifacts", action="store_true", help="Keep successful synthetic outputs for debugging")
    commands.add_parser("clean-smokes", help="Remove only successful synthetic smoke outputs")
    splits = commands.add_parser("check-splits", help="Validate public dev/heldout JSON arrays")
    splits.add_argument("--dev", type=Path, required=True)
    splits.add_argument("--heldout", type=Path, required=True)
    data = commands.add_parser("prepare-gsm8k", help="Prepare public splits and separate private references")
    data.add_argument("--train", type=Path, required=True)
    data.add_argument("--test", type=Path, required=True)
    data.add_argument("--revision", required=True)
    data.add_argument("--dev-count", type=int, default=20)
    data.add_argument("--heldout-count", type=int, default=100)
    generate = commands.add_parser("generate-pool", help="Run a development or frozen GSM8K pool on a CUDA/BF16 server")
    generate.add_argument("--questions", type=Path, required=True)
    generate.add_argument("--config", type=Path)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--allow-download", action="store_true", help="Allow downloading the pinned model if not cached")
    judge = commands.add_parser("judge-pool", help="Prepare Jev requests; only --live sends paid requests")
    judge.add_argument("--questions", type=Path, required=True)
    judge.add_argument("--candidates", type=Path, required=True)
    judge.add_argument("--config", type=Path)
    judge.add_argument("--output", type=Path)
    judge.add_argument("--live", action="store_true")
    judge.add_argument("--env-file", type=Path, help="Explicit private key file; only read for live judging")
    evaluate = commands.add_parser("evaluate-pool", help="Compare selectors offline with private numeric references")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--candidates", type=Path, required=True)
    evaluate.add_argument("--references", type=Path, required=True)
    evaluate.add_argument("--selections", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--config", type=Path)
    from .evaluation.answers import SCORERS
    evaluate.add_argument("--scorer", choices=SCORERS, default="gsm8k_strict_v1")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                "machine": platform.machine(), "local_scaffold_supported": sys.version_info >= (3, 11),
                "optional_packages_installed": {m: importlib.util.find_spec(m) is not None
                    for m in ("torch", "transformers", "datasets")},
                "real_generator_adapter": "official_llada implemented; see reports/weekly/week01.md for server evidence",
                "jev_transport": "Choice HTTP adapter implemented; see reports/weekly/week01.md for live evidence",
                "jev_api_key_present": bool(os.environ.get("JEV_API_KEY")),
                "note": "Package presence does not establish CUDA/BF16/model readiness."}, indent=2))
        elif args.command == "gsm8k500-check":
            from .experiments import frozen
            config, _ = load(args.root / frozen.PROFILE)
            protocol = frozen.check_protocol(args.root, config, args.root / config["datasets"]["questions_path"])
            print(json.dumps({"status": "passed", "protocol": protocol, "question_count": 500,
                "candidates_per_question": 8, "total_candidates": 4000, "scorer": config["evaluation"]["scorer"],
                "gpu_or_api_work_started": False, "full_week1_complete": False}, indent=2))
        elif args.command == "check-config":
            config, pending = load(args.config or args.root / "configs/week01/protocol.json")
            print(json.dumps({"structurally_valid": True, "frozen": config["frozen"],
                              "unresolved_fields": pending, "runtime_readiness_verified": False}, indent=2))
            return int(args.require_resolved and bool(pending))
        elif args.command == "check-splits":
            dev = [Question(**row) for row in json.loads(args.dev.read_text())]
            heldout = [Question(**row) for row in json.loads(args.heldout.read_text())]
            print(json.dumps(validate_splits(dev, heldout), indent=2))
        elif args.command == "prepare-gsm8k":
            from .data.gsm8k import prepare
            print(prepare(args.root, args.train, args.test, args.revision, args.dev_count, args.heldout_count))
        elif args.command == "generate-pool":
            from .experiments.generation import generate_pool
            config, pending = load(args.config or args.root / "configs/week01/pilot.json")
            if config["schema_version"] != 2:
                raise ValueError("generate-pool uses the pilot configuration")
            print(generate_pool(args.root, config, args.questions, args.output or args.root / "artifacts/runs", args.allow_download))
        elif args.command == "judge-pool":
            from .experiments.selection import judge_pools
            from .experiments.selection import load_questions
            config, pending = load(args.config or args.root / "configs/week01/pilot.json")
            from .experiments import frozen
            protocol = (frozen.check_pool(args.root, config, args.questions, args.candidates)
                        if frozen.required(config, load_questions(args.questions)) else None)
            if args.live:
                from .judges.credentials import load_api_key
                load_api_key(args.env_file)
            if args.live and any(q.split == "heldout" for q in load_questions(args.questions)) and not config["frozen"]:
                raise ValueError("Freeze the reviewed protocol before judging held-out questions")
            prompt_path = args.root / config["prompts"]["jev_final"]["path"]
            if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != config["prompts"]["jev_final"]["sha256"]:
                raise ValueError("Jev prompt differs from the recorded configuration")
            print(judge_pools(args.questions, args.candidates, prompt_path,
                args.output or args.root / "artifacts/runs", config["judge"]["model_version"],
                seed=config["runtime"]["master_seed"], live=args.live, protocol=protocol))
        elif args.command == "evaluate-pool":
            from .experiments.selection import evaluate_pools
            from .experiments.selection import load_questions
            config, pending = load(args.config or args.root / "configs/week01/pilot.json")
            if any(q.split == "heldout" for q in load_questions(args.questions)) and not config["frozen"]:
                raise ValueError("Freeze the reviewed protocol before inspecting held-out outcomes")
            from .experiments import frozen
            protocol = (frozen.check_pool(args.root, config, args.questions, args.candidates)
                        if frozen.required(config, load_questions(args.questions)) else None)
            if protocol:
                frozen.check_evaluation(config, args.references, args.scorer, args.selections, protocol)
            print(evaluate_pools(args.questions, args.candidates, args.references,
                args.output or args.root / "artifacts/runs", args.selections, scorer=args.scorer,
                seed=config["runtime"]["master_seed"], protocol=protocol))
        elif args.command == "clean-smokes":
            from .operations.smoke_cleanup import clean_successful_smokes
            print(json.dumps(clean_successful_smokes(args.root), indent=2))
        else:
            path = run(args.root, args.output)
            if args.keep_artifacts:
                print(f"Synthetic smoke check complete (NOT a benchmark result): {path}")
            else:
                from .operations.smoke_cleanup import remove_successful_smoke
                receipt = remove_successful_smoke(path, path.parent)
                print(f"Synthetic smoke passed; removed successful fixture outputs: {receipt['run_id']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
