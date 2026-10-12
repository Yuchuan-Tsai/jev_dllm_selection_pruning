"""Enforce the scoped GSM8K-500 freeze before generation or held-out analysis."""
import hashlib
import json
from pathlib import Path
from ..config import validate
from ..reproducibility import fingerprint, trajectory_seed

PROFILE = "configs/week01/gsm8k500.json"
RECORD = "configs/week01/gsm8k500-freeze-v2.json"
PHASE = "gsm8k500_heldout_selection"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(config, questions):
    return config.get("phase") == PHASE or any(q.split == "heldout" for q in questions)


def check_protocol(root: Path, config: dict, questions_path: Path) -> dict:
    """Read public questions and pins only; never open private references here."""
    from .selection import load_questions
    from ..data.splits import validate_splits
    from ..backends.fast_llada import verify_sources
    if config.get("phase") != PHASE or config.get("frozen") is not True:
        raise ValueError("Held-out runs require the scoped frozen configs/week01/gsm8k500.json profile")
    if validate(config):
        raise ValueError("Frozen GSM8K configuration has unresolved fields")
    record = json.loads((root / RECORD).read_text())
    if record.get("scope") != PHASE or record.get("profile_sha256") != sha256(root / PROFILE):
        raise ValueError("Frozen profile differs from its freeze record")
    if fingerprint(config) != fingerprint(json.loads((root / PROFILE).read_text())):
        raise ValueError("Supplied configuration differs from the frozen GSM8K profile")
    for name, expected in record["public_files_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "private" in relative.parts:
            raise ValueError("Freeze public-file map must contain public project-relative paths only")
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen file changed: {name}; do not silently change the held-out protocol")
    if sha256(questions_path) != config["datasets"]["questions_sha256"]:
        raise ValueError("Use the exact pinned 500-question held-out file, not the development set")
    questions = load_questions(questions_path)
    if len(questions) != 500 or any(q.dataset != "gsm8k" or q.split != "heldout" for q in questions):
        raise ValueError("Require exactly 500 held-out GSM8K questions")
    manifest = json.loads((root / config["datasets"]["manifest_path"]).read_text())
    if [q.question_id for q in questions] != manifest["question_ids"]["heldout"]:
        raise ValueError("Question order/IDs differ from the split manifest")
    dev = load_questions(root / "data/processed/gsm8k-dev.json")
    validate_splits(dev, questions)
    verify_sources(root, config)
    return {"scope": PHASE, "profile_sha256": record["profile_sha256"],
            "freeze_record_sha256": sha256(root / RECORD), "frozen_at_utc": record["frozen_at_utc"]}


def check_pool(root, config, questions_path, candidates_path):
    """Require the verified complete generation archive and its unchanged ordered pools."""
    from .selection import load_pools
    from ..operations.remote import verify_copy, snapshot_digest
    provenance = check_protocol(root, config, questions_path)
    run = candidates_path.parent
    receipt = json.loads((run.parent / ".receipts" / (run.name + ".json")).read_text())
    if receipt.get("digest") != snapshot_digest(receipt["snapshot"]):
        raise ValueError("Archive receipt is inconsistent; repeat results-verify")
    if receipt["snapshot"].get("run_id") != run.name:
        raise ValueError("Archive receipt belongs to a different run")
    verify_copy(run.parent, receipt["snapshot"])
    manifest = json.loads((run / "manifest.json").read_text())
    snapshot = json.loads((run / "config.snapshot.json").read_text())
    if fingerprint(snapshot) != fingerprint(config) or manifest.get("protocol") != provenance:
        raise ValueError("Generation archive does not match this frozen protocol")
    if (manifest.get("status"), manifest.get("question_count"), manifest.get("completed_candidates"),
            manifest.get("questions_sha256")) != ("complete", 500, 4000, config["datasets"]["questions_sha256"]):
        raise ValueError("Require a completed 500-question / 4,000-candidate archive")
    questions, pools = load_pools(questions_path, candidates_path)
    for question in questions:
        pool = pools[question.question_id]
        if len(pool) != 8:
            raise ValueError("Frozen comparison requires eight candidates per question")
        for index, candidate in enumerate(pool):
            if (candidate.candidate_id != f"{question.question_id}/c{index:02}"
                    or candidate.seed != trajectory_seed(config["runtime"]["master_seed"], question.question_id, index)):
                raise ValueError("Candidate order, identity or seed differs from the frozen protocol")
    return provenance


def check_evaluation(config, references_path, scorer, selections_path=None, provenance=None):
    if scorer != config["evaluation"]["scorer"]:
        raise ValueError("Frozen GSM8K-500 evaluation requires gsm8k_units_v3")
    if sha256(references_path) != config["evaluation"]["references_sha256"]:
        raise ValueError("Private references differ from the frozen pin")
    if selections_path:
        manifest = json.loads((selections_path.parent / "manifest.json").read_text())
        if (manifest.get("protocol") != provenance or manifest.get("model") != config["judge"]["model_version"]
                or manifest.get("master_seed") != config["runtime"]["master_seed"]
                or manifest.get("successful_calls") != 500):
            raise ValueError("Jev results do not match the frozen protocol or full coverage")


def check_hardware(config, torch):
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != config["runtime"]["hardware"]):
        raise RuntimeError("Frozen GSM8K-500 profile requires exactly one NVIDIA A100-SXM4-80GB")
