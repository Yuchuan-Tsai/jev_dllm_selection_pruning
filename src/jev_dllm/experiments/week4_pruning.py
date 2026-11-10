"""Partial-draft pruning replay on archived Week 2 trajectories.

The first pilot ranks 16 saved intermediate drafts per question with one Jev
Choice request, retains the top K, and evaluates retained full continuations
against the original complete pool. Cost savings are counterfactual estimates;
this replay does not claim to have stopped a live GPU generation.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

from . import selection, week2_pool
from ..evaluation.answers import numeric_answer, numeric_value
from ..evaluation import week2_labels
from ..judges.credentials import load_api_key
from ..judges.listwise import ListwiseJudge, parse_listwise_response
from ..judges.payloads import partial_request
from ..reproducibility import fingerprint
from ..schema import Candidate, PartialDraft, Question

VERSION = "week4_partial_pruning_replay_v1"
MODEL = "jev-1.13.0"
DEFAULT_POOL = "artifacts/downloads/lab/w2pool-gsm8k-20261026T204300Z-2603fbb3"
DEFAULT_POOLS = {"gsm8k": DEFAULT_POOL,
                 "math": "artifacts/downloads/lab/w2pool-math-20261026T204300Z-10703977"}
PROMPTS = {"gsm8k": "prompts/week04/jev_partial_pruning_gsm8k.txt",
           "math": "prompts/week04/jev_partial_pruning_math.txt"}
MATH_LABELS = "artifacts/runs/w2labels-math-20261028T040147Z-8ed63234"
CODE_FILES = ("src/jev_dllm/experiments/week4_pruning.py", "src/jev_dllm/judges/payloads.py",
              "src/jev_dllm/judges/listwise.py", "src/jev_dllm/judges/credentials.py",
              "src/jev_dllm/evaluation/answers.py", "src/jev_dllm/reproducibility.py",
              "src/jev_dllm/evaluation/week2_labels.py", "scripts/week4_pruning.py",
              "requirements/week4-pruning-cpu.in", PROMPTS["gsm8k"], PROMPTS["math"])
CLAIMS = "artifacts/week4-pruning-claims"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL file: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def _inputs(root: Path, pool: Path, task: str, first: int, checkpoint: int):
    source_task, _, _, _, _, _ = week2_labels.audit_pool(root, pool)
    if source_task != task or task not in PROMPTS:
        raise ValueError("Partial-pruning replay supports gsm8k and math only")
    profile, base, profile_questions = week2_pool.load_profile(root, task)
    questions_path = root / profile["questions"]["path"]
    questions = [Question(**r) for r in json.loads(questions_path.read_text())]
    candidates = rows(pool / "candidates.jsonl")
    costs = rows(pool / "costs.jsonl")
    traces = rows(pool / "trajectories.jsonl")
    if not (len(candidates) == len(costs) == len(traces) == 3200):
        raise ValueError(f"{task} source must contain its complete {len(questions)} x {profile['candidates_per_question']} candidate pool")
    if first < 1 or first > len(questions):
        raise ValueError(f"FIRST must select between 1 and {len(questions)} development questions")
    qids = [q.question_id for q in questions[:first]]
    qmap = {q.question_id: q for q in questions}
    by_c = {r["candidate_id"]: r for r in candidates}
    by_cost = {r["candidate_id"]: r for r in costs}
    by_trace = {r["candidate_id"]: r for r in traces}
    if not (set(by_c) == set(by_cost) == set(by_trace)):
        raise ValueError("Candidate, cost and trajectory IDs do not match")
    if len(by_c) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    profile = json.loads((pool / "pool-profile.json").read_text())
    checkpoints = profile["checkpoint_completed_blocks"]
    if checkpoint not in checkpoints:
        raise ValueError(f"CHECKPOINT must be one of {checkpoints}")
    return qmap, qids, by_c, by_cost, by_trace, profile, base, questions_path


def _math_labels(root: Path, pool: Path) -> tuple[dict, dict]:
    label_dir = root / MATH_LABELS
    manifest = json.loads((label_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("task") != "math":
        raise ValueError("Require the completed Week 2 MATH label run")
    expected = {name: sha(pool / name) for name in ("candidates.jsonl", "costs.jsonl", "trajectories.jsonl")}
    if manifest.get("pool_run") != pool.name or manifest.get("pool_inputs_sha256") != expected:
        raise ValueError("MATH labels do not match this exact candidate pool")
    label_path = label_dir / "evaluation_private/labels.jsonl"
    if manifest.get("labels_sha256") != sha(label_path):
        raise ValueError("Saved MATH labels differ from their manifest")
    label_rows = rows(label_path)
    correct = {r["candidate_id"]: r["correct"] for r in label_rows}
    if len(correct) != 3200 or len(correct) != len(label_rows) or any(type(v) is not bool for v in correct.values()):
        raise ValueError("MATH scorer labels must cover all 3,200 unique candidates")
    hashes = {"labels": sha(label_path), "manifest": sha(label_dir / "manifest.json"),
              "metrics": sha(label_dir / "metrics.json")}
    return correct, hashes


def _tokenizer(root: Path, task: str, allow_download: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError("Tokenizer dependency missing; run make week4-pruning-setup") from None
    _, base, _ = week2_pool.load_profile(root, task)
    g = base["generator"]
    return AutoTokenizer.from_pretrained(g["model_id"], revision=g["tokenizer_revision"],
        trust_remote_code=True, local_files_only=not allow_download)


def _draft_text(tokenizer, trace: dict, checkpoint: int, block_size: int) -> str:
    entry = next((x for x in trace["checkpoints"] if x["completed_blocks"] == checkpoint), None)
    if entry is None:
        raise ValueError(f"Missing checkpoint {checkpoint} for {trace['candidate_id']}")
    # The recorder stores the whole state. Only completed blocks are a real draft;
    # future masked positions are intentionally excluded.
    ids = entry["state_ids"][:checkpoint * block_size]
    ids = [x for x in ids if x != trace["mask_id"]]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _request(question: Question, drafts: list[PartialDraft], instructions: str) -> dict:
    state = partial_request(question, drafts)
    ids = [d.candidate_id for d in drafts]
    return {"model": MODEL, "state": state,
            "questions": {"best_answer": {"type": "choice", "instructions": instructions,
              "criteria": {cid: f"Select candidate {cid} from state.candidates." for cid in ids}}}}


def prepare(root: Path, task: str, pool: Path, first: int = 20, checkpoint: int = 2,
            keep: int = 8, allow_download: bool = False) -> Path:
    if task not in PROMPTS:
        raise ValueError("TASK must be gsm8k or math")
    try:
        pool_name = pool.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError("Pool must be inside this project") from None
    if ".." in Path(pool_name).parts or "private" in Path(pool_name).parts:
        raise ValueError("Pool path must be a public project path")
    qmap, qids, by_c, by_cost, by_trace, profile, base, questions_path = _inputs(root, pool, task, first, checkpoint)
    n = profile["candidates_per_question"]
    if not 2 <= keep < n:
        raise ValueError("KEEP must be at least 2 and less than the 16 candidates per question")
    tokenizer = _tokenizer(root, task, allow_download)
    prompt_path = PROMPTS[task]
    instructions = (root / prompt_path).read_text()
    block_size = base["generator"]["block_size"]
    pool_by_question = {}
    for row in by_c.values():
        pool_by_question.setdefault(row["question_id"], []).append(row)
    planned = []
    for qid in qids:
        q = qmap[qid]
        pool_rows = pool_by_question[qid]
        drafts = []
        for c in pool_rows:
            cid = c["candidate_id"]
            trace = by_trace[cid]
            cp = next(x for x in trace["checkpoints"] if x["completed_blocks"] == checkpoint)
            text = _draft_text(tokenizer, trace, checkpoint, block_size)
            remaining = max(0, by_cost[cid]["trajectory_nfe"] - cp["nfe_before"])
            drafts.append(PartialDraft(qid, cid, text, checkpoint, trace["checkpoints"][-1]["total_blocks"],
                                       remaining, "trajectory_nfe", "fully_predicted"))
        body = _request(q, drafts, instructions)
        # Validate our request shape with the same parser used for live Jev responses.
        expected = set(body["questions"]["best_answer"]["criteria"])
        if len(expected) != n:
            raise ValueError("Partial request does not contain the full candidate pool")
        planned.append({"call_index": len(planned), "question_id": qid,
                        "candidate_ids": [d.candidate_id for d in drafts], "body": body})
    out = selection.new_run(root / "artifacts/runs", f"week4-pruning-{task}-plan")
    (out / "requests.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in planned))
    filenames = ("candidates.jsonl", "costs.jsonl", "trajectories.jsonl", "pool-profile.json", "manifest.json")
    _, label_hashes = _math_labels(root, pool) if task == "math" else ({}, None)
    manifest = {"version": VERSION, "status": "prepared", "task": task, "model": MODEL,
        "pool_dir": pool_name, "pool_hashes": {n: sha(pool / n) for n in filenames},
        "questions_path": questions_path.relative_to(root).as_posix(), "questions_sha256": sha(questions_path),
        "prompt_path": prompt_path, "prompt_sha256": sha(root / prompt_path),
        "implementation_sha256": {f: sha(root / f) for f in CODE_FILES},
        "label_hashes": label_hashes,
        "question_ids": qids, "checkpoint_completed_blocks": checkpoint, "keep": keep,
        "candidates_per_question": n, "planned_calls": len(planned), "paid_calls_sent": 0,
        "requests_sha256": sha(out / "requests.jsonl"), "host_python": platform.python_version(),
        "tokenizer_revision": base["generator"]["tokenizer_revision"],
        "protocol": "Retain the top K candidates by Jev choice probabilities; replay their saved complete continuations for outcome and cost analysis."}
    selection.write_json(out / "plan.json", manifest)
    return out


def check(root: Path, plan_dir: Path):
    p = json.loads((plan_dir / "plan.json").read_text())
    if (p.get("version") != VERSION or p.get("status") != "prepared" or p.get("paid_calls_sent") != 0
        or p.get("host_python") != platform.python_version()
        or p.get("implementation_sha256") != {f: sha(root / f) for f in CODE_FILES}
        or p.get("task") not in PROMPTS or p.get("prompt_path") != PROMPTS.get(p.get("task"))
        or p.get("prompt_path") not in PROMPTS.values()
        or p.get("prompt_sha256") != sha(root / p.get("prompt_path", ""))
        or p.get("questions_sha256") != sha(root / p.get("questions_path", ""))):
        raise ValueError("Plan or code/prompt inputs changed; prepare a new plan")
    pool_name = Path(p.get("pool_dir", ""))
    question_name = Path(p.get("questions_path", ""))
    if (pool_name.is_absolute() or ".." in pool_name.parts or "private" in pool_name.parts
            or question_name.is_absolute() or ".." in question_name.parts or "private" in question_name.parts):
        raise ValueError("Plan pool path must stay inside public project data")
    pool = root / p["pool_dir"]
    if p["pool_hashes"] != {n: sha(pool / n) for n in p["pool_hashes"]}:
        raise ValueError("Archived pool changed since preparation")
    if p["task"] == "math":
        _, label_hashes = _math_labels(root, pool)
        if p.get("label_hashes") != label_hashes:
            raise ValueError("Pinned MATH scoring labels changed")
    if sha(plan_dir / "requests.jsonl") != p["requests_sha256"]:
        raise ValueError("Prepared request file changed")
    calls = rows(plan_dir / "requests.jsonl")
    if len(calls) != p["planned_calls"] or len(calls) != len(p["question_ids"]):
        raise ValueError("Request count differs from the plan")
    ids = set()
    for call in calls:
        body = call["body"]
        expected = set(body["questions"]["best_answer"]["criteria"])
        if (call["question_id"] not in p["question_ids"] or ids.intersection(expected)
                or len(expected) != p["candidates_per_question"]
                or body["state"].get("stage") != "partial"
                or set(call["candidate_ids"]) != expected):
            raise ValueError("Invalid partial-pruning request")
        ids.update(expected)
    return p, calls


def live(root: Path, plan_dir: Path, approved_calls: int, env_file: Path | None = None) -> Path:
    p, calls = check(root, plan_dir)
    if type(approved_calls) is not int or approved_calls != len(calls):
        raise ValueError(f"Approve exactly {len(calls)} Jev calls to run this plan")
    load_api_key(env_file)
    claims = root / CLAIMS
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"version": VERSION, "requests": p["requests_sha256"]})
    out = plan_dir / "live"
    out.mkdir(parents=True, exist_ok=False)
    try:
        with (claims / (claim_id + ".json")).open("x") as f:
            f.write(json.dumps({"plan": str(plan_dir), "output": str(out)}) + "\n")
    except FileExistsError:
        raise ValueError("This paid request plan has already been claimed; do not replay it") from None
    judge = ListwiseJudge()
    manifest = {"status": "running", "attempted_calls": 0, "successful_calls": 0,
                "plan_sha256": sha(plan_dir / "plan.json"), "requests_sha256": p["requests_sha256"],
                "api_cost_usd": None, "automatic_retries": 0}
    selection.write_json(out / "manifest.json", manifest)
    try:
        with (out / "results.jsonl").open("x") as f:
            for call in calls:
                manifest["attempted_calls"] += 1
                selection.write_json(out / "manifest.json", manifest)
                result = judge.select(call["body"])
                parse_listwise_response(result["raw_response"], call["body"])
                record = {"call_index": call["call_index"], "question_id": call["question_id"],
                          "candidate_ids": call["candidate_ids"], **result}
                record.pop("raw_response", None)
                f.write(json.dumps(record, allow_nan=False) + "\n")
                f.flush(); os.fsync(f.fileno())
                manifest["successful_calls"] += 1
                selection.write_json(out / "manifest.json", manifest)
        manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
        raise
    finally:
        selection.write_json(out / "manifest.json", manifest)
    return out


def analyze(root: Path, plan_dir: Path) -> Path:
    p, calls = check(root, plan_dir)
    live_dir = plan_dir / "live"
    lm = json.loads((live_dir / "manifest.json").read_text())
    if lm.get("status") != "complete" or lm.get("successful_calls") != len(calls):
        raise ValueError("Run all planned Jev requests before analysis")
    results = rows(live_dir / "results.jsonl")
    if len(results) != len(calls):
        raise ValueError("Incomplete Jev results")
    pool = root / p["pool_dir"]
    candidates = {r["candidate_id"]: Candidate(**r) for r in rows(pool / "candidates.jsonl")}
    costs = {r["candidate_id"]: r for r in rows(pool / "costs.jsonl")}
    traces = {r["candidate_id"]: r for r in rows(pool / "trajectories.jsonl")}
    if p["task"] == "gsm8k":
        _, _, questions = week2_pool.load_profile(root, "gsm8k")
        refs = {qid: row["answer"] for qid, row in week2_labels.references(root, "gsm8k", questions).items()}
        correct = {cid: bool(c.completed and not c.truncated and
                     numeric_answer(c.text, "gsm8k_units_v3") == numeric_value(refs[c.question_id]))
                   for cid, c in candidates.items()}
    elif p["task"] == "math":
        correct, _ = _math_labels(root, pool)
        if set(correct) != set(candidates):
            raise ValueError("MATH labels do not cover the full source candidate pool")
    else:
        raise ValueError("Unsupported pruning task")
    chosen_rows, full_oracle, kept_oracle, first_acc = [], [], [], []
    saved_gpu, baseline_gpu, saved_nfe, baseline_nfe = 0., 0., 0, 0
    for call, result in zip(calls, results):
        if result["question_id"] != call["question_id"]:
            raise ValueError("Jev result order differs from the plan")
        by_prob = result["probabilities"]
        ranked = sorted(call["candidate_ids"], key=lambda cid: (-by_prob[cid], cid))
        kept = ranked[:p["keep"]]
        dropped = ranked[p["keep"]:]
        qid = call["question_id"]
        full_oracle.append(any(correct[cid] for cid in call["candidate_ids"]))
        kept_oracle.append(any(correct[cid] for cid in kept))
        first_acc.append(correct[kept[0]])
        for cid in call["candidate_ids"]:
            baseline_gpu += costs[cid]["generation_gpu_seconds"]
            baseline_nfe += costs[cid]["trajectory_nfe"]
        for cid in dropped:
            cp = next(x for x in traces[cid]["checkpoints"] if x["completed_blocks"] == p["checkpoint_completed_blocks"])
            elapsed = cp["gpu_seconds_before"]
            if elapsed is None:
                raise ValueError("Checkpoint GPU timings missing; cannot estimate savings")
            saved_gpu += max(0., costs[cid]["generation_gpu_seconds"] - elapsed)
            saved_nfe += max(0, costs[cid]["trajectory_nfe"] - cp["nfe_before"])
        chosen_rows.append({"question_id": qid, "ranked_candidate_ids": ranked,
                            "retained_candidate_ids": kept, "pruned_candidate_ids": dropped})
    count = len(calls)
    out = selection.new_run(root / "artifacts/runs", f"week4-pruning-{p['task']}-analysis")
    metrics = {"version": VERSION, "task": p["task"], "questions": count,
        "checkpoint_completed_blocks": p["checkpoint_completed_blocks"],
        "retained_per_question": p["keep"], "candidates_per_question": p["candidates_per_question"],
        "retained_pool_oracle_accuracy": sum(kept_oracle) / count,
        "full_pool_oracle_accuracy": sum(full_oracle) / count,
        "oracle_accuracy_change_percentage_points": 100 * (sum(kept_oracle) - sum(full_oracle)) / count,
        "top_ranked_candidate_accuracy": sum(first_acc) / count,
        "counterfactual_compute": {"estimated_gpu_seconds_saved": saved_gpu,
            "estimated_fraction_of_full_pool_gpu_seconds_saved": saved_gpu / baseline_gpu if baseline_gpu else None,
            "estimated_nfe_saved": saved_nfe,
            "estimated_fraction_of_full_pool_nfe_saved": saved_nfe / baseline_nfe if baseline_nfe else None,
            "baseline_full_pool_gpu_seconds": baseline_gpu, "baseline_full_pool_nfe": baseline_nfe,
            "note": "Replay estimate from recorded full runs and checkpoint times; does not measure live stopping, resume overhead, or GPU contention."},
        "jev_plan": plan_dir.name, "jevs_scored_calls": len(results),
        "scorer": "gsm8k_units_v3" if p["task"] == "gsm8k" else "math500_math_verify_v1",
        "note": "Development pilot only. Retained-pool oracle is an upper bound, not final selector accuracy."}
    selection.write_json(out / "metrics.json", metrics)
    (out / "selections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen_rows))
    selection.write_json(out / "manifest.json", {"status": "complete", "plan": plan_dir.name,
        "pool_run": Path(p["pool_dir"]).name, "task": p["task"],
        "metrics_sha256": sha(out / "metrics.json"), "api_calls": 0,
        "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    return out


def compare(root: Path, analysis_dirs: list[Path]) -> Path:
    if len(analysis_dirs) != 3:
        raise ValueError("Compare exactly three checkpoint analyses")
    entries = []
    for directory in analysis_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        if (manifest.get("status") != "complete"
                or manifest.get("metrics_sha256") != sha(metrics_path)
                or metrics.get("task") not in PROMPTS
                or manifest.get("task") not in (None, metrics.get("task"))):
            raise ValueError(f"Not a complete, unchanged pruning analysis: {directory}")
        plan_dir = root / "artifacts/runs" / manifest["plan"]
        plan = json.loads((plan_dir / "plan.json").read_text())
        expected_code = {f: sha(root / f) for f in CODE_FILES}
        if plan.get("implementation_sha256") == expected_code:
            plan, calls = check(root, plan_dir)
        else:
            # Completed runs remain comparable after this module changes; verify
            # their public pool, request plan and live result evidence directly.
            if not (plan.get("task") in PROMPTS
                    and metrics.get("task") == plan.get("task")
                    and plan.get("checkpoint_completed_blocks") in {2, 4, 6, 8, 12}):
                raise ValueError("Historical plan task or checkpoint is not supported")
            pool = root / plan["pool_dir"]
            if plan.get("pool_hashes") != {name: sha(pool / name) for name in plan.get("pool_hashes", {})}:
                raise ValueError("Historical GSM8K source pool differs from its recorded hashes")
            if sha(plan_dir / "requests.jsonl") != plan.get("requests_sha256"):
                raise ValueError("Historical GSM8K request plan changed")
            live_manifest_path = plan_dir / "live/manifest.json"
            live_manifest = json.loads(live_manifest_path.read_text())
            if (live_manifest.get("status") != "complete"
                    or live_manifest.get("successful_calls") != plan.get("planned_calls")
                    or live_manifest.get("requests_sha256") != plan.get("requests_sha256")
                    or metrics.get("jevs_scored_calls") != plan.get("planned_calls")):
                raise ValueError("Historical GSM8K checkpoint-2 Jev run is not complete")
            calls = rows(plan_dir / "requests.jsonl")
        if (len(calls) != 200 or metrics.get("questions") != 200
                or metrics.get("checkpoint_completed_blocks") != plan["checkpoint_completed_blocks"]
                or metrics.get("retained_per_question") != plan["keep"]):
            raise ValueError("Each comparison analysis must cover the same 200-question development set")
        entries.append((plan, metrics, directory))
    entries.sort(key=lambda item: item[0]["checkpoint_completed_blocks"])
    checkpoints = [item[0]["checkpoint_completed_blocks"] for item in entries]
    task = entries[0][0]["task"]
    expected_checkpoints = {"gsm8k": [2, 4, 6], "math": [4, 8, 12]}[task]
    if any(item[0]["task"] != task for item in entries) or checkpoints != expected_checkpoints:
        raise ValueError(f"Expected {task} checkpoints {expected_checkpoints}, got {checkpoints}")
    first = entries[0][0]
    for plan, metrics, _ in entries[1:]:
        if (plan["pool_hashes"] != first["pool_hashes"]
                or plan["question_ids"] != first["question_ids"]
                or plan["keep"] != first["keep"]
                or plan.get("label_hashes") != first.get("label_hashes")):
            raise ValueError("Checkpoint plans must use the same pool, labels, questions and KEEP count")
    table = []
    for plan, metrics, directory in entries:
        compute = metrics["counterfactual_compute"]
        table.append({"checkpoint": plan["checkpoint_completed_blocks"],
            "approx_tokens_seen": plan["checkpoint_completed_blocks"] * 32,
            "retained_pool_oracle_accuracy": metrics["retained_pool_oracle_accuracy"],
            "full_pool_oracle_accuracy": metrics["full_pool_oracle_accuracy"],
            "oracle_change_percentage_points": metrics["oracle_accuracy_change_percentage_points"],
            "top_ranked_candidate_accuracy": metrics["top_ranked_candidate_accuracy"],
            "estimated_gpu_seconds_saved_fraction": compute["estimated_fraction_of_full_pool_gpu_seconds_saved"],
            "estimated_nfe_saved_fraction": compute["estimated_fraction_of_full_pool_nfe_saved"],
            "analysis_directory": directory.name})
    out = selection.new_run(root / "artifacts/runs", f"week4-pruning-{task}-comparison")
    task_label = "GSM8K" if task == "gsm8k" else "MATH"
    record = {"status": "complete", "task": task, "question_count": 200,
        "pool_run": Path(first["pool_dir"]).name, "retained_candidates_per_question": first["keep"],
        "checkpoints": table,
        "note": f"Retrospective replay on the same {task_label} development pool; oracle accuracy is an upper bound and compute savings are counterfactual estimates, not live GPU measurements."}
    selection.write_json(out / "comparison.json", record)
    lines = [f"# {task_label} partial-draft checkpoint comparison", "",
        f"Same 200 {task_label} development questions and 16 LLaDA candidates per question. Jev ranks drafts at each saved checkpoint; this comparison retains the top 8. This is retrospective replay, not live pruning.", "",
        "| Completed blocks | Approx. token positions | Retained-pool oracle accuracy | Full-pool oracle accuracy | Change (pp) | Top-ranked accuracy | Est. GPU seconds saved | Est. NFE saved |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['checkpoint']} | {row['approx_tokens_seen']} | {row['retained_pool_oracle_accuracy']:.1%} | {row['full_pool_oracle_accuracy']:.1%} | {row['oracle_change_percentage_points']:+.1f} | {row['top_ranked_candidate_accuracy']:.1%} | {row['estimated_gpu_seconds_saved_fraction']:.1%} | {row['estimated_nfe_saved_fraction']:.1%} |")
    lines += ["", "Oracle accuracy means at least one correct completed answer remains in the retained candidates; it is not final Jev selection accuracy. Savings are estimated from the archived full-run timing/NFE records and do not include resume overhead.", ""]
    (out / "comparison.md").write_text("\n".join(lines))
    selection.write_json(out / "manifest.json", {"status": "complete", "comparison_sha256": sha(out / "comparison.json"),
        "report_sha256": sha(out / "comparison.md"), "analysis_runs": [item[2].name for item in entries],
        "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    return out
