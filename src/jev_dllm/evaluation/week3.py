"""Week 3 selector comparison on verified Week 2 development pools (local, no paid calls).

Inputs: one complete Week 2 label run, plus any score sources that exist (a complete
Jev live run, Qwen judge and PRM runs). Every source is checked against the exact
pool it scored. Selectors that need a missing source are skipped and listed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import math500, week2_labels
from .answers import numeric_answer
from .week3_metrics import calibration, compare, within_question_ranking
from ..experiments import week2_jev
from ..experiments.selection import new_run, write_json
from ..judges import open_models
from ..operations import remote
from ..selection import week3 as select
from ..selection.subsets import SIZES, nested_subsets

SEED = 20261008
VOTING_TASKS = ("gsm8k", "math")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_labels(root: Path, labels_dir: Path):
    m = json.loads((labels_dir / "manifest.json").read_text())
    if m.get("status") != "complete" or sha256(labels_dir / "evaluation_private/labels.jsonl") != m.get("labels_sha256"):
        raise ValueError("Require a complete, unchanged Week 2 label run")
    pool_dir = root / "artifacts/downloads/lab" / m["pool_run"]
    task, _, questions, pools, _, traces = week2_labels.audit_pool(root, pool_dir)
    if task != m["task"] or {name: sha256(pool_dir / name) for name in m["pool_inputs_sha256"]} != m["pool_inputs_sha256"]:
        raise ValueError("Label run does not belong to this pool copy")
    correct = {r["candidate_id"]: r["correct"] for r in rows(labels_dir / "evaluation_private/labels.jsonl")}
    labeled = [q for q in questions if all(c.candidate_id in correct for c in pools[q.question_id])]
    return task, pool_dir, labeled, pools, traces, correct


def verified_run(run: Path) -> dict:
    """A run copied by results-pull and checked by results-verify."""
    receipt = json.loads((run.parent / ".receipts" / (run.name + ".json")).read_text())
    if receipt["snapshot"].get("run_id") != run.name or receipt["digest"] != remote.snapshot_digest(receipt["snapshot"]):
        raise ValueError(f"{run.name}: invalid receipt; run results-verify first")
    remote.verify_copy(run.parent, receipt["snapshot"])
    return json.loads((run / "manifest.json").read_text())


def open_scores(root: Path, run: Path, kind: str, task: str, pool_dir: Path) -> dict:
    m = verified_run(run)
    if (m.get("kind") != kind or m.get("task") != task or m.get("status") != "complete"
            or m.get("limited_to_questions") is not None
            or m.get("pool_candidates_sha256") != sha256(pool_dir / "candidates.jsonl")
            or m.get("profile_sha256") != sha256(root / open_models.PROFILE)):
        raise ValueError(f"{run.name} is not a complete, full-pool {kind} run on this pool and profile")
    key = "p_yes" if kind == "judge" else "score"
    return {"scores": {r["candidate_id"]: r[key] for r in rows(run / "results.jsonl")},
            "seconds": m["total_seconds_including_load"], "run": run.name}


def jev_scores(root: Path, plan_dir: Path, task: str, pool_dir: Path) -> dict:
    plan, _ = week2_jev.check(root, plan_dir)
    # A run that stopped and was finished by week2_jev_continuation lives in live-continued/.
    output = next((plan_dir / name for name in ("live", "live-continued")
                   if (plan_dir / name / "manifest.json").exists()
                   and json.loads((plan_dir / name / "manifest.json").read_text()).get("status") == "complete"), None)
    if output is None:
        raise ValueError("Require a complete Jev live run (or a complete continuation of one)")
    live = json.loads((output / "manifest.json").read_text())
    if (plan["task"] != task or live.get("results_sha256") != sha256(output / "results.jsonl")
            or plan["candidates_sha256"] != sha256(pool_dir / "candidates.jsonl")):
        raise ValueError("Require a complete Jev live run planned on this exact pool")
    listwise, correct_p, consistent_p = {}, {}, {}
    for r in rows(output / "results.jsonl"):
        if r["kind"] == "listwise":
            listwise[(r["question_id"], r["subset_size"])] = {"choice": r["candidate_id"], "probabilities": r["probabilities"]}
        else:
            correct_p[r["candidate_id"]] = r["correct_probability"]
            consistent_p[r["candidate_id"]] = r["consistent_probability"]
    return {"listwise": listwise, "correct": correct_p, "consistent": consistent_p,
            "questions": set(plan["question_ids"]), "usage": live.get("usage"), "plan": plan_dir.name,
            "output": output.name, "continued_from": live.get("continued_from")}


def answer_keys(root, task, questions, pools, math_python):
    """Reference-free answer keys for voting: GSM8K numeric values; MATH Math-Verify groups."""
    if task == "gsm8k":
        return {c.candidate_id: (numeric_answer(c.text, "gsm8k_units_v3") if c.completed and not c.truncated else None)
                for q in questions for c in pools[q.question_id]}
    if task != "math":
        return None
    if math_python is None:
        raise ValueError("MATH voting needs the Math-Verify interpreter (--math-python)")
    policy = math500.read(root / math500.POLICY)
    refs = {r["question_id"]: r["answer"] for r in json.loads((root / "data/private/math-w2dev-references.json").read_text())}

    def group(q):
        candidates = [{k: getattr(c, k) for k in ("candidate_id", "text", "completed", "truncated")} for c in pools[q.question_id]]
        # The worker forms these groups before it reads the reference; only the public part is used.
        record = math500.worker(root, math_python, "question", policy,
                                question={"question_id": q.question_id, "candidates": candidates, "reference": refs[q.question_id]})
        return q.question_id, record["public_selection"]["groups"]

    keys = {}
    with ThreadPoolExecutor(max_workers=policy["workers"]) as executor:
        for qid, groups in executor.map(group, questions):
            keys.update(select.prefix_group_keys([c.candidate_id for c in pools[qid]], groups))
    return keys


def selections_for(n, questions, pools, keys, sources, task):
    """Every available selector's pick on subset N of each question."""
    picks = {}
    subsets = {q.question_id: nested_subsets([c.candidate_id for c in pools[q.question_id]], (n,))[n] for q in questions}
    def add(name, choose):
        picks[name] = {qid: choose(qid, ids) for qid, ids in subsets.items()}
    add("first", lambda qid, ids: select.first(ids))
    add("seeded_random", lambda qid, ids: select.seeded_random(qid, ids, SEED))
    if keys is not None:
        add("majority", lambda qid, ids: select.majority(ids, keys))
    for variant in ("mean_probability", "min_probability", "mean_log_probability"):
        scores = sources["confidence"][variant]
        add(f"confidence_{variant}", lambda qid, ids, s=scores: select.highest(ids, s))
    jev = sources.get("jev")
    if jev:
        add("jev_listwise", lambda qid, ids: jev["listwise"][(qid, n)]["choice"])
        add("jev_pointwise", lambda qid, ids: select.highest(ids, jev["correct"]))
        if keys is not None:
            add("jev_weighted_vote", lambda qid, ids: select.weighted_vote(ids, keys, jev["correct"]))
    for name, source in (("qwen_judge", sources.get("judge")), ("prm_min", sources.get("prm"))):
        if source:
            add(name, lambda qid, ids, s=source["scores"]: select.highest(ids, s))
    return subsets, picks


def analyze(root: Path, labels_dir: Path, math_python: Path | None = None, jev_plan: Path | None = None,
            judge_run: Path | None = None, prm_run: Path | None = None) -> Path:
    task, pool_dir, questions, pools, traces, correct = load_labels(root, labels_dir)
    sources = {"confidence": {v: {t["candidate_id"]: t["confidence"][v] for t in traces}
                              for v in ("mean_probability", "min_probability", "mean_log_probability")}}
    skipped, provenance = [], {"labels_run": labels_dir.name, "pool_run": pool_dir.name}
    if jev_plan:
        jev = jev_scores(root, jev_plan, task, pool_dir)
        if jev["questions"] >= {q.question_id for q in questions}:
            sources["jev"] = jev
            provenance.update(jev_plan=jev["plan"], jev_output=jev["output"], jev_continued_from=jev["continued_from"])
        else:
            skipped.append(f"Jev covers {len(jev['questions'])} of {len(questions)} questions; Jev selectors need full coverage")
    else:
        skipped.append("No Jev live run supplied")
    for kind, run in (("judge", judge_run), ("prm", prm_run)):
        if run:
            sources[kind] = open_scores(root, run, kind, task, pool_dir)
            provenance[f"{kind}_run"] = run.name
        else:
            skipped.append(f"No {kind} run supplied")
    keys = answer_keys(root, task, questions, pools, math_python) if task in VOTING_TASKS else None
    out = new_run(root / "artifacts/runs", f"w3analysis-{task}")
    (out / "evaluation_private").mkdir()
    sizes = tuple(n for n in SIZES if n <= min(len(pools[q.question_id]) for q in questions))
    by_n, private = {}, []
    for n in sizes:
        subsets, picks = selections_for(n, questions, pools, keys, sources, task)
        by_n[str(n)] = compare(picks, subsets, correct, SEED)
        for name, chosen in picks.items():
            private += [{"subset_size": n, "selector": name, "question_id": qid, "candidate_id": cid, "correct": correct[cid]}
                        for qid, cid in chosen.items()]
    full = {q.question_id: [c.candidate_id for c in pools[q.question_id]] for q in questions}
    ranking = {f"confidence_{v}": within_question_ranking(s, full, correct) for v, s in sources["confidence"].items()}
    calibrated = {}
    if "jev" in sources:
        ranking["jev_pointwise_correct"] = within_question_ranking(sources["jev"]["correct"], full, correct)
        ranking["jev_pointwise_consistent"] = within_question_ranking(sources["jev"]["consistent"], full, correct)
        largest = {cid: p for (qid, n), r in sources["jev"]["listwise"].items() if n == sizes[-1] for cid, p in r["probabilities"].items()}
        ranking[f"jev_listwise_probability_n{sizes[-1]}"] = within_question_ranking(largest, full, correct)
        calibrated["jev_pointwise_correct"] = calibration(sources["jev"]["correct"], correct)
    for kind, name in (("judge", "qwen_judge"), ("prm", "prm_min")):
        if kind in sources:
            ranking[name] = within_question_ranking(sources[kind]["scores"], full, correct)
    if "judge" in sources:
        calibrated["qwen_judge_p_yes"] = calibration(sources["judge"]["scores"], correct)
    costs = {"jev_usage": sources["jev"]["usage"] if "jev" in sources else None,
             "qwen_judge_gpu_seconds": sources["judge"]["seconds"] if "judge" in sources else None,
             "prm_gpu_seconds": sources["prm"]["seconds"] if "prm" in sources else None,
             "note": "Majority voting and dLLM confidence add no model calls; confidence was recorded during generation."}
    metrics = {"task": task, "questions": len(questions), "subset_rule": "first N candidates in pool order",
               "by_subset_size": by_n, "within_question_ranking_full_pool": ranking, "calibration": calibrated,
               "costs": costs, "skipped": skipped, "provenance": provenance, "seed": SEED,
               "note": "Development pools only. Oracle uses private labels and is not a selector; ties go to pool order."}
    write_json(out / "metrics.json", metrics)
    with (out / "evaluation_private/selections.jsonl").open("x") as stream:
        for row in private:
            stream.write(json.dumps(row) + "\n")
    (out / "summary.md").write_text(summary_table(metrics))
    write_json(out / "manifest.json", {"status": "complete", "task": task, "provenance": provenance,
                                       "metrics_sha256": sha256(out / "metrics.json"),
                                       "finished_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0})
    return out


def summary_table(metrics: dict) -> str:
    sizes = list(metrics["by_subset_size"])
    names = list(metrics["by_subset_size"][sizes[-1]]["selectors"])
    lines = [f"Task: {metrics['task']} ({metrics['questions']} questions)", "",
             "| Selector | " + " | ".join(f"N={n}" for n in sizes) + f" | Gap closed N={sizes[-1]} |",
             "|---|" + "---|" * (len(sizes) + 1)]
    for name in names:
        cells = [f"{metrics['by_subset_size'][n]['selectors'][name]['accuracy']:.1%}" for n in sizes]
        gap = metrics["by_subset_size"][sizes[-1]]["selectors"][name]["gap_closed"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {'n/a' if gap is None else f'{gap:.2f}'} |")
    for label, key in (("Mean single-sample", "mean_single_sample_accuracy"), ("Oracle", "oracle_accuracy")):
        lines.append(f"| {label} | " + " | ".join(f"{metrics['by_subset_size'][n][key]:.1%}" for n in sizes) + " | |")
    lines += ["", "| Score | Mean within-question AUROC | Mixed questions |", "|---|---|---|"]
    for name, r in metrics["within_question_ranking_full_pool"].items():
        value = r["mean_within_question_auroc"]
        lines.append(f"| {name} | {'n/a' if value is None else f'{value:.3f}'} | {r['mixed_questions']} |")
    if metrics["skipped"]:
        lines += ["", "Skipped: " + "; ".join(metrics["skipped"])]
    return "\n".join(lines) + "\n"
