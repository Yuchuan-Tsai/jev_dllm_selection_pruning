"""Public-only HumanEval Jev preparation, explicit paid execution, private scoring."""
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import random

from . import collection, frozen
from .selection import load_pools, new_run, paired_summary, write_json
from ..evaluation import humaneval_extracted as extraction
from ..judges.typesafe import TypeSafeJudge, build_request, parse_response
from ..reproducibility import fingerprint, trajectory_seed
from ..selection.baselines import random_choice

VERSION = "humaneval_extracted_jev_v1"
MODEL = "jev-1.13"
SEED = 20261008
PROMPT = "prompts/jev_humaneval.txt"
FILES = ("src/jev_dllm/experiments/humaneval_jev.py", "scripts/judge_humaneval.py", PROMPT,
         "src/jev_dllm/judges/typesafe.py", "src/jev_dllm/judges/payloads.py",
         "src/jev_dllm/experiments/selection.py", "src/jev_dllm/reproducibility.py") + extraction.FILES


def read_rows(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete final record; retain the evidence")
    return [json.loads(line) for line in raw.splitlines()]


def implementation(root):
    return {p: frozen.sha256(root / p) for p in FILES}


def public_inputs(root, archive):
    audit = collection.audit(root, archive)
    if audit["task"] != "humaneval" or audit["status"] != "complete":
        raise ValueError("Require a complete, verified HumanEval archive")
    questions, pools = load_pools(root / "data/processed/humaneval-heldout.json", archive / "candidates.jsonl")
    if len(questions) != 164 or any(len(p) != 8 for p in pools.values()):
        raise ValueError("Require all 164 tasks and eight candidates per task")
    return questions, pools


def request_rows(questions, pools, prompt):
    """Never read private scores; retain all candidate IDs, including rejected extraction."""
    requests, transforms = [], []
    for q in questions:
        cleaned = []
        for c in pools[q.question_id]:
            e, record = extraction.extraction_record(c)
            # Failed extraction stays in the pool as original text. No repairs or filtering.
            text = c.text if e.source is None else e.source
            cleaned.append(replace(c, text=text))
            transforms.append(record)
        random.Random(trajectory_seed(SEED, q.question_id, 0)).shuffle(cleaned)
        requests.append({"question_id": q.question_id, "body": build_request(q, cleaned, prompt, MODEL)})
    return requests, transforms


def prepare(root, archive):
    questions, pools = public_inputs(root, archive)
    requests, transforms = request_rows(questions, pools, (root / PROMPT).read_text())
    out = new_run(root / "artifacts/runs", "humaneval-jev-plan")
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    write_json(out / "transformations.json", transforms)
    plan = {"version": VERSION, "model": MODEL, "seed": SEED,
            "source_run": str(archive.resolve()), "source_candidates_sha256": frozen.sha256(archive / "candidates.jsonl"),
            "question_count": len(questions), "candidate_count": len(transforms),
            "requests_sha256": frozen.sha256(out / "requests.jsonl"),
            "transformations_sha256": frozen.sha256(out / "transformations.json"),
            "implementation_sha256": implementation(root), "extraction_host_python": platform.python_version(),
            "analysis_type": "post_hoc_extraction_selection", "automatic_retries": 0,
            "rejected_extraction_policy": "Retain original response; private scoring counts extraction failure incorrect.",
            "paid_calls_sent": 0}
    write_json(out / "plan.json", plan)
    return out


def inspect(root, plan_dir):
    p = json.loads((plan_dir / "plan.json").read_text())
    if (p["version"] != VERSION or p["model"] != MODEL or p["seed"] != SEED
            or p["automatic_retries"] != 0 or p["implementation_sha256"] != implementation(root)
            or p["extraction_host_python"] != platform.python_version()):
        raise ValueError("Prepared policy/runtime changed; do not execute this plan")
    archive = Path(p["source_run"])
    questions, pools = public_inputs(root, archive)
    requests, transforms = request_rows(questions, pools, (root / PROMPT).read_text())
    if (p["source_candidates_sha256"] != frozen.sha256(archive / "candidates.jsonl")
            or p["requests_sha256"] != frozen.sha256(plan_dir / "requests.jsonl")
            or p["transformations_sha256"] != frozen.sha256(plan_dir / "transformations.json")
            or read_rows(plan_dir / "requests.jsonl") != requests
            or json.loads((plan_dir / "transformations.json").read_text()) != transforms
            or p["question_count"] != len(questions) or p["candidate_count"] != len(transforms)):
        raise ValueError("Prepared public inputs changed")
    return p, requests, questions, pools


def execute(root, plan_dir, approved_calls, client=None):
    p, requests, _, _ = inspect(root, plan_dir)
    if type(approved_calls) is not int or approved_calls != len(requests):
        raise ValueError("Approve exactly the number of paid requests in this plan")
    # Same public batch cannot be replayed by preparing a second directory.
    claims = root / "artifacts/humaneval-jev-claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"source": p["source_candidates_sha256"], "requests": p["requests_sha256"]})
    out = plan_dir / "live"
    with (claims / (claim_id + ".json")).open("x") as claim:
        claim.write(json.dumps({"plan": str(plan_dir), "output": str(out)}) + "\n")
    out.mkdir(exist_ok=False)
    m = {"status": "running", "plan_sha256": frozen.sha256(plan_dir / "plan.json"),
         "attempted_calls": 0, "successful_calls": 0, "approved_calls": approved_calls,
         "api_cost_usd": None, "automatic_retries": 0}
    write_json(out / "manifest.json", m)
    print(f"Jev output directory: {out}", flush=True)
    client = client or TypeSafeJudge()
    try:
        with (out / "selections.jsonl").open("x") as stream:
            for row in requests:
                m["attempted_calls"] += 1
                write_json(out / "manifest.json", m)
                result = client.select(row["body"])
                parsed = parse_response(result["raw_response"], row["body"])
                if any(result.get(k) != v for k, v in parsed.items()):
                    raise ValueError("Saved selection differs from validated response")
                stream.write(json.dumps({"question_id": row["question_id"], **result}, allow_nan=False) + "\n")
                stream.flush()
                m["successful_calls"] += 1
                write_json(out / "manifest.json", m)
                print(f"HumanEval Jev: {m['successful_calls']}/{len(requests)}", flush=True)
        inspect(root, plan_dir)
        m.update(status="complete", selections_sha256=frozen.sha256(out / "selections.jsonl"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        if "row" in locals():
            m["failed_question_id"] = row["question_id"]
        if hasattr(exc, "raw_response"):
            write_json(out / "failed_response.json", {"question_id": row["question_id"],
                "raw_response_json": json.dumps(exc.raw_response, ensure_ascii=False), "request_seconds": exc.request_seconds})
        raise
    finally:
        write_json(out / "manifest.json", m)
    return out


def score_labels(root, archive, score_run, questions, pools):
    """Private phase only: verify and reuse the already executed extraction tests."""
    m = json.loads((score_run / "manifest.json").read_text())
    if (m["status"] != "complete" or Path(m["source_run"]).resolve() != archive.resolve()
            or m["policy"] != extraction.load_policy(root)
            or m["runtime_profile"] != extraction.strict.load_profile(root)
            or m["implementation_sha256"] != {p: frozen.sha256(root / p) for p in extraction.FILES}
            or m["host_python_for_static_extraction"] != platform.python_version()
            or m["results_sha256"] != frozen.sha256(score_run / "candidate-results.jsonl")):
        raise ValueError("Require completed extraction scores of this exact archive and policy")
    for field in ("input_sha256", "strict_evidence_sha256"):
        for name, digest in m[field].items():
            if frozen.sha256(Path(name)) != digest:
                raise ValueError("Original scoring evidence changed")
    rows = read_rows(score_run / "candidate-results.jsonl")
    base = extraction.summarize(questions, pools, rows, SEED)
    candidates = [c for q in questions for c in pools[q.question_id]]
    if m["completed_candidates"] != len(candidates):
        raise ValueError("Incomplete private scoring")
    for c, row in zip(candidates, rows):
        e, record = extraction.extraction_record(c)
        if any(row.get(k) != v for k, v in record.items()):
            raise ValueError("Private scoring source differs from judge source")
        if e.source is None and row["status"] != "extraction_failure":
            raise ValueError("Rejected extraction cannot receive a passing label")
    return {r["candidate_id"]: r["status"] == "passed" for r in rows}, base


def evaluate(root, plan_dir, score_run):
    p, requests, questions, pools = inspect(root, plan_dir)
    live = plan_dir / "live"
    m = json.loads((live / "manifest.json").read_text())
    selections = read_rows(live / "selections.jsonl")
    if (m["status"] != "complete" or m["plan_sha256"] != frozen.sha256(plan_dir / "plan.json")
            or m["attempted_calls"] != len(questions) or m["successful_calls"] != len(questions)
            or m["approved_calls"] != len(questions) or len(selections) != len(questions)
            or m["selections_sha256"] != frozen.sha256(live / "selections.jsonl")):
        raise ValueError("Require a complete, unchanged Jev batch")
    for result, request in zip(selections, requests):
        parsed = parse_response(result["raw_response"], request["body"])
        if result["question_id"] != request["question_id"] or any(result.get(k) != v for k,v in parsed.items()):
            raise ValueError("Selection does not match its public request and raw response")
        seconds = result["request_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid request latency")
    labels, base = score_labels(root, Path(p["source_run"]), score_run, questions, pools)
    chosen = [labels[r["candidate_id"]] for r in selections]
    first = [labels[pools[q.question_id][0].candidate_id] for q in questions]
    random_labels = [labels[random_choice(pools[q.question_id], trajectory_seed(SEED, q.question_id, 1))] for q in questions]
    base.update(jev_evaluated=True, analysis_type=p["analysis_type"],
        jev={"correct": sum(chosen), "total": len(chosen), "accuracy": sum(chosen)/len(chosen)},
        jev_vs_first_sample=paired_summary(chosen, first, SEED),
        jev_vs_seeded_random=paired_summary(chosen, random_labels, SEED),
        attempted_calls=m["attempted_calls"], api_cost_usd=None,
        request_seconds_sum=sum(r["request_seconds"] for r in selections),
        usage={key: sum(r["usage"][key] for r in selections) for key in ("input_tokens", "output_tokens")},
        note="Same saved candidate pools and extracted-code labels for every selector; extraction failures retained. Post hoc reanalysis, not an untouched held-out benchmark claim.")
    out = new_run(root / "artifacts/runs", "humaneval-jev-evaluation")
    write_json(out / "metrics.json", base)
    write_json(out / "manifest.json", {"status": "complete", "plan": str(plan_dir), "score_run": str(score_run),
        "evidence_sha256": {str(f): frozen.sha256(f) for f in
            (plan_dir / "plan.json", live / "manifest.json", live / "selections.jsonl",
             score_run / "manifest.json", score_run / "candidate-results.jsonl")},
        "implementation_sha256": implementation(root), "new_api_calls": 0})
    print(json.dumps(base, indent=2))
    return out
