"""Pinned ProcessBench subset diagnostic with public-only Jev requests."""
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from ..experiments.selection import new_run, write_json
from ..judges.typesafe import TypeSafeJudge

CODE_REVISION = "e8024636bcabdf8bd514440551b531d3f90dd18b"
CODE_URL = "https://raw.githubusercontent.com/QwenLM/ProcessBench/" + CODE_REVISION + "/code/run_eval.py"
CODE_SHA256 = "66d09fc7a3d20d46f166d4bba4e04835f3d7473e1534b6908cd9045aefc7b704"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_request(row: dict, instructions: str, model: str) -> dict:
    if set(row) != {"question_id", "subset", "problem", "steps"}:
        raise ValueError("ProcessBench judge inputs must contain public fields only")
    if (not isinstance(row["question_id"], str) or not row["question_id"]
            or row["subset"] not in {"gsm8k", "math"}
            or not isinstance(row["problem"], str) or not row["problem"].strip()
            or not isinstance(row["steps"], list) or not 1 <= len(row["steps"]) <= 254
            or any(not isinstance(s, str) or not s.strip() for s in row["steps"])):
        raise ValueError("Invalid ProcessBench public input")
    if not instructions.strip() or not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Pin the Jev model and diagnostic instructions")
    criteria = {"-1": "Every step is correct; no erroneous step was found."}
    criteria.update({str(i): f"Step {i} is the earliest erroneous step." for i in range(len(row["steps"]))})
    return {"model": model, "state": {"problem": row["problem"],
            "steps": [{"index": i, "text": s} for i, s in enumerate(row["steps"])]},
            "questions": {"best_answer": {"type": "choice", "instructions": instructions, "criteria": criteria}}}


def official_metrics(labels: list[int], predictions: list[int | None]) -> dict:
    """Same class-conditional exact-index rates and harmonic mean as pinned run_eval.py."""
    if not labels or len(labels) != len(predictions) or any(type(x) is not int or x < -1 for x in labels):
        raise ValueError("Expected aligned nonempty first-error labels and predictions")
    if any(x is not None and (type(x) is not int or x < -1) for x in predictions):
        raise ValueError("Invalid predicted first-error index")
    error_matches = [p == y for y, p in zip(labels, predictions) if y != -1]
    correct_matches = [p == y for y, p in zip(labels, predictions) if y == -1]
    a = mean(error_matches) * 100 if error_matches else None
    b = mean(correct_matches) * 100 if correct_matches else None
    f1 = None if a is None or b is None else (2 * a * b / (a + b) if a + b else 0.0)
    return {"count": len(labels), "error_count": len(error_matches), "correct_count": len(correct_matches),
            "error_exact_index_accuracy_percent": a, "correct_solution_accuracy_percent": b,
            "f1_percent": f1,
            "undefined_policy": "Missing class: null; both class accuracies zero: F1=0."}


def judge_subset(root: Path, output: Path, config_path: Path, live: bool = False, judge=None) -> Path:
    config = json.loads(config_path.read_text())
    manifest_path = root / "data/manifests/processbench-week01.json"
    if sha(manifest_path) != config["data_manifest_sha256"]:
        raise ValueError("ProcessBench manifest differs from the diagnostic configuration")
    data_manifest = json.loads(manifest_path.read_text())
    if data_manifest["official_code_revision"] != CODE_REVISION:
        raise ValueError("Unexpected official ProcessBench scorer revision")
    prompt_path = root / config["prompt"]["path"]
    if sha(prompt_path) != config["prompt"]["sha256"]:
        raise ValueError("Diagnostic prompt differs from configuration")
    requests, inputs = [], {}
    for subset in ("gsm8k", "math"):
        relative = f"data/processed/processbench-{subset}-diagnostic.json"
        path = root / relative
        if sha(path) != data_manifest["files"][relative]:
            raise ValueError("Public ProcessBench input hash mismatch")
        rows = json.loads(path.read_text())
        if [r["question_id"] for r in rows] != data_manifest["subsets"][subset]["question_ids"]:
            raise ValueError("Diagnostic IDs differ from the pinned subset")
        for row in rows:
            if row["subset"] != subset:
                raise ValueError("Diagnostic subset mismatch")
            requests.append({"question_id": row["question_id"], "subset": subset,
                             "body": build_request(row, prompt_path.read_text(), config["model"])})
        inputs[relative] = sha(path)
    if len({r["question_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate diagnostic IDs")
    # No label file is opened by this function.
    run = new_run(output, "processbench-live" if live else "processbench-dry-run")
    write_json(run / "config.snapshot.json", config)
    (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    manifest = {"mode": "processbench_live" if live else "processbench_dry_run",
                "status": "prepared", "model": config["model"], "question_count": len(requests),
                "attempted_calls": 0, "successful_calls": 0, "input_sha256": inputs,
                "requests_sha256": sha(run / "requests.jsonl"),
                "prompt_sha256": sha(prompt_path), "data_manifest_sha256": sha(manifest_path),
                "official_code_revision": CODE_REVISION, "api_cost_usd": None if live else 0,
                "subset_only": True, "semantic_overlap_review_complete": False,
                "instruction_note": "Choice adaptation of the official earliest-error task, not its free-text critique prompt."}
    write_json(run / "manifest.json", manifest)
    if not live:
        return run
    client = judge or TypeSafeJudge()
    try:
        with (run / "predictions.jsonl").open("w") as stream:
            for row in requests:
                manifest["attempted_calls"] += 1
                write_json(run / "manifest.json", manifest)
                result = client.select(row["body"])
                prediction = int(result["candidate_id"])
                if str(prediction) not in row["body"]["questions"]["best_answer"]["criteria"]:
                    raise ValueError("Predicted index does not belong to the supplied solution")
                stream.write(json.dumps({"question_id": row["question_id"], "subset": row["subset"],
                                         "prediction": prediction, **result}) + "\n")
                stream.flush()
                manifest["successful_calls"] += 1
    except BaseException as exc:
        if hasattr(exc, "raw_response"):
            write_json(run / "failed_response.json", {"question_id": row["question_id"],
                       "raw_response_json": json.dumps(exc.raw_response, ensure_ascii=False),
                           "request_seconds": exc.request_seconds})
        manifest.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         "error_type": type(exc).__name__, "failed_question_id": row["question_id"]})
        write_json(run / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    manifest["predictions_sha256"] = sha(run / "predictions.jsonl")
    write_json(run / "manifest.json", manifest)
    return run


def evaluate_subset(root: Path, source: Path, output: Path) -> Path:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("mode") != "processbench_live" or manifest.get("status") != "complete":
        raise ValueError("Evaluate only a complete live diagnostic; retain failed runs without dropping rows")
    if sha(source / "predictions.jsonl") != manifest["predictions_sha256"] or sha(source / "requests.jsonl") != manifest["requests_sha256"]:
        raise ValueError("Diagnostic outputs differ from their recorded hashes")
    data_path = root / "data/manifests/processbench-week01.json"
    if sha(data_path) != manifest["data_manifest_sha256"]:
        raise ValueError("Diagnostic dataset provenance differs")
    data_manifest = json.loads(data_path.read_text())
    raw = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()]
    predictions = {r["question_id"]: r for r in raw}
    if len(predictions) != len(raw):
        raise ValueError("Duplicate diagnostic predictions")
    all_ids, private, scores, label_hashes = set(), [], {}, {}
    for subset in ("gsm8k", "math"):
        public_path = root / f"data/processed/processbench-{subset}-diagnostic.json"
        if sha(public_path) != manifest["input_sha256"][str(public_path.relative_to(root))]:
            raise ValueError("Diagnostic public input changed")
        steps = {r["question_id"]: len(r["steps"]) for r in json.loads(public_path.read_text())}
        relative = f"data/private/processbench-{subset}-labels.json"
        label_path = root / relative
        if sha(label_path) != data_manifest["files"][relative]:
            raise ValueError("Private diagnostic label hash mismatch")
        labels = json.loads(label_path.read_text())
        ids = [r["question_id"] for r in labels]
        if len(set(ids)) != len(ids) or set(ids) != set(steps):
            raise ValueError("Private diagnostic labels do not match public inputs")
        all_ids.update(ids)
        predicted = []
        for r in labels:
            qid, label = r["question_id"], r["label"]
            if type(label) is not int or not -1 <= label < steps[qid]:
                raise ValueError("Invalid reference first-error index")
            result = predictions[qid]
            pred = result["prediction"]
            if result["subset"] != subset or type(pred) is not int or not -1 <= pred < steps[qid]:
                raise ValueError("Invalid prediction or subset")
            predicted.append(pred)
            private.append({"question_id": qid, "label": label, "prediction": pred, "match": pred == label})
        scores[subset] = official_metrics([r["label"] for r in labels], predicted)
        label_hashes[relative] = sha(label_path)
    if set(predictions) != all_ids:
        raise ValueError("Diagnostic predictions do not cover exactly the fixed subsets")
    run = new_run(output, "processbench-evaluation")
    write_json(run / "metrics.json", {"subset_results": scores, "subset_only": True,
              "official_code_revision": CODE_REVISION, "official_code_sha256": CODE_SHA256,
              "official_code_url": CODE_URL, "no_selection_improvement_claim": True,
              "semantic_overlap_review_complete": False})
    (run / "evaluation_private").mkdir()
    write_json(run / "evaluation_private/question_results.json", private)
    write_json(run / "manifest.json", {"mode": "processbench_evaluation", "source_run": source.name,
              "source_manifest_sha256": sha(source / "manifest.json"),
              "predictions_sha256": sha(source / "predictions.jsonl"), "private_label_sha256": label_hashes})
    return run
