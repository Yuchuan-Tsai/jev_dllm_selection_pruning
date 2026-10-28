"""Week 2 Jev on development pools: offline prepare, offline check, approved paid live run.

Per question: one listwise Choice request per nested subset size, plus one pointwise
request (two noul targets) per candidate in the largest subset. Nothing here reads
references or labels, and only live() can send a paid request.
"""
import json
import math
from pathlib import Path
import platform
import time

from . import frozen
from .selection import load_pools, new_run, write_json
from ..judges.credentials import load_api_key
from ..judges.listwise import build_listwise_request, nested_subsets, parse_listwise_response
from ..judges.pointwise import PointwiseJudge, build_pointwise_request, parse_pointwise_response
from ..judges.listwise import ROUNDING_STEP, ListwiseJudge
from ..reproducibility import fingerprint

# v2 accepts a listwise choice one two-decimal rounding step below the top probability, flagged
# rounding_tie; v1 rejected it. A new version also means new claims, so v1 batches stay as evidence.
VERSION = "week2_jev_v2"
MODEL = "jev-1.13.0"
SEED = 20261008
SUBSET_SIZES = (2, 4, 8, 16)
CONSISTENT_PROMPT = "prompts/week02/jev_pointwise_consistent.txt"
TASKS = {
    "gsm8k": {"listwise": "prompts/jev_final.txt", "correct": "prompts/week02/jev_pointwise_correct_math.txt"},
    "math": {"listwise": "prompts/jev_final.txt", "correct": "prompts/week02/jev_pointwise_correct_math.txt"},
    "mbpp": {"listwise": "prompts/jev_humaneval.txt", "correct": "prompts/week02/jev_pointwise_correct_code.txt"},
}
FILES = ("src/jev_dllm/experiments/week2_jev.py", "src/jev_dllm/judges/pointwise.py",
         "src/jev_dllm/judges/listwise.py", "src/jev_dllm/judges/typesafe.py",
         "src/jev_dllm/judges/payloads.py", "src/jev_dllm/judges/credentials.py",
         "src/jev_dllm/experiments/selection.py", "src/jev_dllm/selection/baselines.py",
         "src/jev_dllm/reproducibility.py", "src/jev_dllm/schema.py")
CLAIMS = "artifacts/week2-jev-claims"


def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete final record; retain the evidence")
    return [json.loads(line) for line in raw.splitlines()]


def implementation(root: Path) -> dict:
    for name in FILES:
        if any((root / part).is_symlink() for part in (Path(name), *Path(name).parents)):
            raise ValueError("Jev implementation must not contain symlinks")
    return {p: frozen.sha256(root / p) for p in FILES}


def prompt_paths(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown Week 2 task: {task}; use one of {sorted(TASKS)}")
    return {**TASKS[task], "consistent": CONSISTENT_PROMPT}


def prompt_texts(root: Path, task: str) -> dict:
    texts = {role: (root / path).read_text() for role, path in prompt_paths(task).items()}
    if any(not text.strip() for text in texts.values()):
        raise ValueError("Jev instructions cannot be empty")
    return texts


def public_inputs(pool_dir: Path, questions_path: Path, question_ids: list[str]) -> tuple[list, dict]:
    """Public questions and candidates only, in question_ids order."""
    if (not question_ids or len(set(question_ids)) != len(question_ids)
            or any(not isinstance(q, str) or not q for q in question_ids)):
        raise ValueError("Question IDs must be a nonempty list of unique strings")
    questions, pools = load_pools(questions_path, pool_dir / "candidates.jsonl")
    by_id = {q.question_id: q for q in questions}
    missing = [q for q in question_ids if q not in by_id]
    if missing:
        raise ValueError(f"Question IDs absent from the public questions file: {missing}")
    selected = [by_id[q] for q in question_ids]
    if any(q.split != "dev" for q in selected):
        raise ValueError("Week 2 Jev runs use development questions only")
    return selected, {q.question_id: pools[q.question_id] for q in selected}


def project_path(root: Path, path: Path) -> str:
    """Record inputs relative to the project root so a copied checkout re-verifies a plan."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("Week 2 Jev inputs must be inside the project root") from None
    if "private" in relative.parts or not relative.parts:
        raise ValueError("Week 2 Jev inputs must be public project files")
    return relative.as_posix()


def recorded_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "private" in relative.parts:
        raise ValueError("Plan input paths must be public and project-relative")
    return root / relative


def file_order(question_id: str, candidate_ids: list[str]) -> list[str]:
    return list(candidate_ids)


def candidate_orders(pools: dict, sizes: tuple[int, ...], candidate_order=None) -> dict:
    """candidate_order(question_id, pool IDs in file order) -> ordered IDs; subsets are prefixes."""
    order_fn = candidate_order or file_order
    orders = {}
    for question_id, pool in pools.items():
        ids = [c.candidate_id for c in pool]
        order = list(order_fn(question_id, list(ids)))
        if (len(set(order)) != len(order) or not set(order) <= set(ids)
                or any(not isinstance(cid, str) for cid in order)):
            raise ValueError(f"Candidate order for {question_id} must list unique IDs from its pool")
        nested_subsets(order, sizes)
        orders[question_id] = order
    return orders


def request_rows(questions: list, pools: dict, orders: dict, sizes: tuple[int, ...],
                 prompts: dict, seed: int, model: str) -> list[dict]:
    rows = []
    for q in questions:
        by_id = {c.candidate_id: c for c in pools[q.question_id]}
        subsets = nested_subsets(orders[q.question_id], sizes)
        for n, ids in subsets.items():
            rows.append({"kind": "listwise", "question_id": q.question_id, "subset_size": n,
                         "subset_ids": ids, "body": build_listwise_request(
                             q, [by_id[cid] for cid in ids], prompts["listwise"], seed, model)})
        # Pointwise scores are per candidate, so the largest subset covers every N.
        for cid in subsets[sizes[-1]]:
            rows.append({"kind": "pointwise", "question_id": q.question_id, "candidate_id": cid,
                         "body": build_pointwise_request(q, by_id[cid], prompts["correct"],
                                                         prompts["consistent"], model)})
    return [{"call_index": i, **row} for i, row in enumerate(rows)]


def call_counts(rows: list[dict]) -> dict:
    listwise = sum(r["kind"] == "listwise" for r in rows)
    return {"listwise": listwise, "pointwise": len(rows) - listwise, "total": len(rows)}


def prepare(root: Path, task: str, pool_dir: Path, questions_path: Path, question_ids: list[str],
            sizes: tuple[int, ...] = SUBSET_SIZES, candidate_order=None) -> Path:
    """Offline: freeze every request and the exact paid-call count. Sends nothing."""
    sizes = tuple(sizes)
    pool_name, questions_name = project_path(root, pool_dir), project_path(root, questions_path)
    prompts = prompt_texts(root, task)
    questions, pools = public_inputs(pool_dir, questions_path, question_ids)
    orders = candidate_orders(pools, sizes, candidate_order)
    rows = request_rows(questions, pools, orders, sizes, prompts, SEED, MODEL)
    out = new_run(root / "artifacts/runs", f"week2-jev-{task}-plan")
    (out / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    plan = {"version": VERSION, "task": task, "model": MODEL, "seed": SEED,
            "subset_sizes": list(sizes), "question_ids": list(question_ids),
            "candidate_orders": orders,
            "candidate_order_source": getattr(candidate_order, "__qualname__", None) or "pool_file_order",
            "pool_dir": pool_name,
            "candidates_sha256": frozen.sha256(pool_dir / "candidates.jsonl"),
            "questions_path": questions_name, "questions_sha256": frozen.sha256(questions_path),
            "prompts_sha256": {path: frozen.sha256(root / path) for path in prompt_paths(task).values()},
            "implementation_sha256": implementation(root), "host_python": platform.python_version(),
            "requests_sha256": frozen.sha256(out / "requests.jsonl"),
            "planned_calls": call_counts(rows),
            "calls_per_question": {"listwise": len(sizes), "pointwise": sizes[-1]},
            "automatic_retries": 0, "paid_calls_sent": 0, "correctness_scored": False,
            "listwise_choice_tolerance": {"rounding_step": ROUNDING_STEP, "requires_two_decimal_probabilities": True,
                                          "flag": "rounding_tie"},
            "candidate_policy": "Original IDs and raw text retained unchanged, including truncated or incomplete answers; no labels or references."}
    write_json(out / "plan.json", plan)
    return out


def check(root: Path, plan_dir: Path) -> tuple[dict, list[dict]]:
    """Offline: re-derive every request from the recorded inputs and compare hashes."""
    p = json.loads((plan_dir / "plan.json").read_text())
    if (p["version"] != VERSION or p["model"] != MODEL or p["seed"] != SEED
            or p["automatic_retries"] != 0 or p["paid_calls_sent"] != 0
            or p["correctness_scored"] is not False
            or p["implementation_sha256"] != implementation(root)
            or p["host_python"] != platform.python_version()):
        raise ValueError("Prepared policy/runtime changed; do not execute this plan")
    task = p["task"]
    if p["prompts_sha256"] != {path: frozen.sha256(root / path) for path in prompt_paths(task).values()}:
        raise ValueError("Jev instructions changed since preparation")
    pool_dir, questions_path = recorded_path(root, p["pool_dir"]), recorded_path(root, p["questions_path"])
    if (p["candidates_sha256"] != frozen.sha256(pool_dir / "candidates.jsonl")
            or p["questions_sha256"] != frozen.sha256(questions_path)):
        raise ValueError("Prepared public inputs changed")
    sizes = tuple(p["subset_sizes"])
    questions, pools = public_inputs(pool_dir, questions_path, p["question_ids"])
    if set(p["candidate_orders"]) != set(pools):
        raise ValueError("Recorded candidate orders do not match the planned questions")
    orders = candidate_orders(pools, sizes, lambda qid, ids: p["candidate_orders"][qid])
    rows = request_rows(questions, pools, orders, sizes, prompt_texts(root, task), SEED, MODEL)
    if (p["requests_sha256"] != frozen.sha256(plan_dir / "requests.jsonl")
            or read_rows(plan_dir / "requests.jsonl") != rows
            or p["planned_calls"] != call_counts(rows)):
        raise ValueError("Prepared requests differ from their recorded inputs")
    return p, rows


def validate_result(result: dict, row: dict) -> dict:
    parse = parse_listwise_response if row["kind"] == "listwise" else parse_pointwise_response
    parsed = parse(result["raw_response"], row["body"])
    if any(result.get(k) != v for k, v in parsed.items()):
        raise ValueError("Saved Jev result differs from its validated response")
    seconds = result["request_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid request latency")
    return parsed


def usage_summary(records: list[dict]) -> dict:
    summary = {}
    for kind in ("listwise", "pointwise"):
        rows = [r for r in records if r["kind"] == kind]
        seconds = [r["request_seconds"] for r in rows]
        summary[kind] = {"calls": len(rows),
                         "input_tokens": sum(r["usage"]["input_tokens"] for r in rows),
                         "output_tokens": sum(r["usage"]["output_tokens"] for r in rows),
                         "request_seconds_sum": sum(seconds),
                         "request_seconds_mean": sum(seconds) / len(seconds) if seconds else None,
                         "request_seconds_max": max(seconds) if seconds else None}
    summary["listwise"]["rounding_ties"] = sum(bool(r.get("rounding_tie")) for r in records if r["kind"] == "listwise")
    return summary


def live(root: Path, plan_dir: Path, approved_calls: int, env_file: Path | None = None,
         listwise_judge=None, pointwise_judge=None) -> Path:
    """Paid: sends only when approved_calls equals the planned count; stops on the first error."""
    p, rows = check(root, plan_dir)
    if type(approved_calls) is not int or approved_calls != p["planned_calls"]["total"] or approved_calls != len(rows):
        raise ValueError(f"Approve exactly the {len(rows)} paid requests in this plan")
    if listwise_judge is None or pointwise_judge is None:
        # The key is read into the environment only; it is never printed or recorded.
        load_api_key(env_file)
    listwise_judge = listwise_judge or ListwiseJudge()
    pointwise_judge = pointwise_judge or PointwiseJudge()
    # One claim per request batch: a second launch of this plan, or of an identical
    # batch prepared in another directory, is refused before any request is sent.
    claims = root / CLAIMS
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = fingerprint({"version": VERSION, "requests": p["requests_sha256"]})
    out = plan_dir / "live"
    try:
        with (claims / (claim_id + ".json")).open("x") as claim:
            claim.write(json.dumps({"plan": str(plan_dir), "output": str(out)}) + "\n")
    except FileExistsError:
        raise ValueError("This Jev batch was already launched; prepare and approve a new plan") from None
    out.mkdir(exist_ok=False)
    m = {"status": "running", "task": p["task"], "plan_sha256": frozen.sha256(plan_dir / "plan.json"),
         "attempted_calls": 0, "successful_calls": 0, "approved_calls": approved_calls,
         "api_cost_usd": None, "automatic_retries": 0}
    write_json(out / "manifest.json", m)
    print(f"Jev output directory: {out}", flush=True)
    records = []
    try:
        with (out / "results.jsonl").open("x") as stream:
            for row in rows:
                m["attempted_calls"] += 1
                write_json(out / "manifest.json", m)
                started = time.perf_counter()
                if row["kind"] == "listwise":
                    result = listwise_judge.select(row["body"])
                else:
                    result = pointwise_judge.judge(row["body"])
                wall = time.perf_counter() - started
                try:
                    validate_result(result, row)
                except (ValueError, KeyError, TypeError) as exc:
                    exc.raw_response = result.get("raw_response") if isinstance(result, dict) else None
                    exc.request_seconds = result.get("request_seconds") if isinstance(result, dict) else None
                    raise
                key = {"subset_size": row["subset_size"]} if row["kind"] == "listwise" else {"candidate_id": row["candidate_id"]}
                record = {"call_index": row["call_index"], "kind": row["kind"],
                          "question_id": row["question_id"], **key, **result, "wall_seconds": wall}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                m["successful_calls"] += 1
                write_json(out / "manifest.json", m)
                print(f"Week 2 Jev {p['task']}: {m['successful_calls']}/{len(rows)}", flush=True)
        check(root, plan_dir)
        m.update(status="complete", results_sha256=frozen.sha256(out / "results.jsonl"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                 error_type=type(exc).__name__)
        if "row" in locals():
            m["failed_call_index"] = row["call_index"]
        if hasattr(exc, "raw_response"):
            write_json(out / "failed_response.json", {"call_index": row["call_index"],
                       "raw_response_json": json.dumps(exc.raw_response, ensure_ascii=False),
                       "request_seconds": exc.request_seconds})
        raise
    finally:
        m["usage"] = usage_summary(records)
        write_json(out / "manifest.json", m)
    return out
