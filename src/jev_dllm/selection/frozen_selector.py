"""Decision Point A: the frozen final selector (Jev pointwise, highest `correct`) and its checks.

The record pins the selection rule, Jev model and protocol, prompts and implementation by SHA-256, and
names the development analyses it was chosen on. Held-out runs must pass `check` before selecting."""
import json
from pathlib import Path
from ..experiments import frozen, week2_jev
from . import week3

RECORD = "configs/week04/selector-freeze-v1.json"
SCOPE = "final_selector_decision_point_a"
SELECTOR = "jev_pointwise"
# Held-out benchmark -> the development task whose pool chose the selector and whose prompts it uses.
BENCHMARKS = {"gsm8k500": "gsm8k", "math500": "math", "humaneval": "mbpp"}
FILES = tuple(sorted(set(week2_jev.FILES) | {"src/jev_dllm/selection/week3.py",
                                              "src/jev_dllm/selection/frozen_selector.py"}))


def prompts(benchmark: str) -> dict:
    """Prompt paths the frozen selector uses for one held-out benchmark."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown held-out benchmark: {benchmark}; use one of {sorted(BENCHMARKS)}")
    paths = week2_jev.prompt_paths(BENCHMARKS[benchmark])
    return {"correct": paths["correct"], "consistent": paths["consistent"]}


def pinned_paths() -> list[str]:
    return sorted(set(FILES) | {p for b in BENCHMARKS for p in prompts(b).values()})


def select(candidate_ids: list[str], correct: dict) -> str:
    """Highest `correct` probability; ties go to the earliest candidate; a missing score ranks last."""
    return week3.highest(candidate_ids, correct)


def check(root: Path) -> dict:
    """Offline: the record, the code constants and every pinned file agree; no labels or API."""
    record = json.loads((root / RECORD).read_text())
    rule = record.get("selector", {})
    if (record.get("scope") != SCOPE or rule.get("name") != SELECTOR or rule.get("interface") != "pointwise"
            or rule.get("score") != "correct" or record.get("held_out_used_for_choice") is not False):
        raise ValueError("Selector freeze record has an unexpected scope or rule")
    jev = record.get("jev", {})
    if (jev.get("model"), jev.get("protocol")) != (week2_jev.MODEL, week2_jev.VERSION):
        raise ValueError("Jev model or protocol differs from the frozen selector")
    if record.get("benchmarks") != {b: {"development_task": t, **prompts(b)} for b, t in BENCHMARKS.items()}:
        raise ValueError("Benchmark-to-prompt map differs from the frozen selector")
    if set(record.get("files_sha256", {})) != set(pinned_paths()):
        raise ValueError("Selector freeze pins an unexpected file list")
    for name, expected in record["files_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Selector freeze must pin project-relative paths only")
        if any((root / p).is_symlink() for p in (relative, *relative.parents)):
            raise ValueError("Pinned selector paths must not contain symlinks")
        if frozen.sha256(root / relative) != expected:
            raise ValueError(f"Frozen selector file changed: {name}; write a new versioned record instead")
    return record


def verify_evidence(root: Path, record: dict) -> dict:
    """Where the Week 3 analyses are present locally, confirm the recorded result they support."""
    found = {}
    for task, item in record["evidence"]["analyses"].items():
        run = root / "artifacts/runs" / item["run"]
        if not (run / "metrics.json").exists():
            found[task] = "not available locally"
            continue
        if frozen.sha256(run / "metrics.json") != item["metrics_sha256"]:
            raise ValueError(f"Week 3 analysis {item['run']} differs from the one the freeze cites")
        selectors = json.loads((run / "metrics.json").read_text())["by_subset_size"][
            str(record["evidence"]["subset_size"])]["selectors"]
        best = max(s["accuracy"] for s in selectors.values())
        if selectors[SELECTOR]["accuracy"] != item["accuracy"] or item["accuracy"] != best:
            raise ValueError(f"{task}: the cited analysis no longer shows {SELECTOR} as most accurate")
        found[task] = "verified"
    return found
