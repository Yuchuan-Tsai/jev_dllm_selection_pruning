"""Week 3 open-model judges on Week 2 pools: Qwen2.5-7B-Instruct and Qwen2.5-Math-PRM-7B.

Pure helpers (prompts, step splitting, aggregation, probabilities) need no GPU; model
code imports torch and transformers lazily from the separate judge environment.
Judges see only the public question and raw candidate text, never labels.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from ..experiments import week2_pool
from ..experiments.selection import new_run, write_json

PROFILE = "configs/week03/open-judges.json"
KINDS = ("judge", "prm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_profile(root: Path) -> dict:
    profile = json.loads((root / PROFILE).read_text())
    if profile.get("scope") != "week03_open_model_judges":
        raise ValueError("Unexpected open-judge profile")
    if sha256(root / profile["requirements"]["path"]) != profile["requirements"]["sha256"]:
        raise ValueError("Judge requirements differ from the profile pin")
    for kind in KINDS:
        if not re.fullmatch(r"[0-9a-f]{40}", profile[kind]["revision"]):
            raise ValueError("Open judges require immutable model revisions")
    for spec in profile["judge"]["prompts"].values():
        if sha256(root / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Judge prompt differs from its pin: {spec['path']}")
    return profile


def check_environment(root: Path) -> dict:
    """Installed judge packages must equal the pinned requirement versions (and the pinned torch)."""
    profile = load_profile(root)
    pins = {"torch": profile["requirements"]["torch"]}
    for line in (root / profile["requirements"]["path"]).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, version = line.split("==")
            pins[name.strip()] = version.strip()
    changed = []
    for name, version in pins.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            changed.append({"package": name, "expected": version, "installed": installed})
    if changed:
        raise RuntimeError(f"Judge environment differs from its pins: {changed}")
    return {"status": "passed", "pinned_packages": len(pins), "python": platform.python_version()}


def judge_prompt(template: str, question: str, candidate: str) -> str:
    # Plain replacement: braces inside questions or code must stay literal.
    return template.replace("{question}", question).replace("{candidate}", candidate)


def split_steps(text: str) -> list[str]:
    """Steps = blank-line-separated paragraphs of the raw answer; never empty."""
    steps = [s.strip() for s in re.split(r"\n\s*\n", text.strip()) if s.strip()]
    return steps or [text.strip() or "(empty answer)"]


def aggregate(step_scores: list[float], rule: str) -> float:
    if not step_scores:
        raise ValueError("No step scores to aggregate")
    if rule == "min":
        return min(step_scores)
    if rule == "last":
        return step_scores[-1]
    if rule == "product":
        return math.prod(step_scores)
    raise ValueError(f"Unknown aggregation rule: {rule}")


def yes_probability(logit_yes: float, logit_no: float) -> float:
    """Two-way softmax of the Yes and No next-token logits."""
    return 1.0 / (1.0 + math.exp(logit_no - logit_yes))


def server_pool(root: Path, run: Path):
    """The server's own complete pool run, checked against the current Week 2 profile."""
    manifest = json.loads((run / "manifest.json").read_text())
    task = manifest.get("task")
    profile, base, questions = week2_pool.load_profile(root, task)
    if (manifest.get("mode") != week2_pool.SCOPE or manifest.get("status") != "complete"
            or (run / "pool-profile.json").read_bytes() != (root / week2_pool.profile_path(task)).read_bytes()):
        raise ValueError("Require a complete Week 2 pool run made with the current pool profile")
    rows = week2_pool.read_rows(run / "candidates.jsonl")
    schedule = week2_pool.plan(questions, profile["candidates_per_question"], base["runtime"]["master_seed"])
    if [(r["candidate_id"], r["seed"]) for r in rows] != [(cid, seed) for _, cid, seed in schedule]:
        raise ValueError("Pool candidates do not follow the planned order and seeds")
    prompts = {q.question_id: q.prompt for q in questions}
    return task, [(prompts[r["question_id"]], r) for r in rows]


class QwenJudge:
    def __init__(self, spec: dict, root: Path, task: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.spec = torch, spec
        self.template = (root / spec["prompts"][task]["path"]).read_text()
        options = {"revision": spec["revision"], "local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **options)
        self.model = AutoModelForCausalLM.from_pretrained(spec["model_id"], torch_dtype=torch.bfloat16, **options).to("cuda").eval()
        ids = [self.tokenizer.encode(spec[k], add_special_tokens=False) for k in ("yes_token", "no_token")]
        if any(len(i) != 1 for i in ids):
            raise ValueError("Yes/No must each be a single token for this tokenizer")
        self.yes, self.no = ids[0][0], ids[1][0]

    def score(self, question: str, candidate: str) -> dict:
        messages = [{"role": "user", "content": judge_prompt(self.template, question, candidate)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] > self.spec["max_input_tokens"]:
            raise ValueError("Judge input exceeds the pinned length limit")
        with self.torch.inference_mode():
            logits = self.model(ids.to("cuda")).logits[0, -1].float()
        logit_yes, logit_no = logits[self.yes].item(), logits[self.no].item()
        return {"p_yes": yes_probability(logit_yes, logit_no), "logit_yes": logit_yes, "logit_no": logit_no,
                "input_tokens": ids.shape[1]}


class ProcessRewardModel:
    def __init__(self, spec: dict, root: Path, task: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.spec = torch, spec
        # trust_remote_code runs the model repository's own code, pinned by the revision above.
        options = {"revision": spec["revision"], "local_files_only": True, "trust_remote_code": True}
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **options)
        self.model = AutoModel.from_pretrained(spec["model_id"], torch_dtype=torch.bfloat16, **options).to("cuda").eval()
        self.separator = self.tokenizer.encode(spec["step_separator"], add_special_tokens=False)
        if len(self.separator) != 1:
            raise ValueError("The step separator must be one token")

    def score(self, question: str, candidate: str) -> dict:
        steps = split_steps(candidate)
        sep = self.spec["step_separator"]
        messages = [{"role": "system", "content": self.spec["system"]}, {"role": "user", "content": question},
                    {"role": "assistant", "content": sep.join(steps) + sep}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = self.tokenizer.encode(text, return_tensors="pt")
        if ids.shape[1] > self.spec["max_input_tokens"]:
            raise ValueError("PRM input exceeds the pinned length limit")
        with self.torch.inference_mode():
            logits = self.model(input_ids=ids.to("cuda"))[0][0].float()
        mask = ids[0].to("cuda") == self.separator[0]
        # Model card: positive-class probability at every step separator.
        step_scores = self.torch.softmax(logits[mask], dim=-1)[:, 1].tolist()
        if len(step_scores) != len(steps):
            raise ValueError("Step separators and step scores disagree")
        return {"step_scores": step_scores, "score": aggregate(step_scores, self.spec["aggregation"]),
                "steps": len(steps), "input_tokens": ids.shape[1]}


def fetch(root: Path, kind: str) -> dict:
    from huggingface_hub import snapshot_download
    spec = load_profile(root)[kind]
    path = snapshot_download(spec["model_id"], revision=spec["revision"],
                             allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt", "tokenizer*", "merges.txt", "vocab.json"])
    return {"model_id": spec["model_id"], "revision": spec["revision"], "cached_model": path}


def run(root: Path, kind: str, pool_run: Path, limit: int | None = None, scorer_factory=None) -> Path:
    """Score every candidate of one server pool run; limit=K scores only the first K questions (smoke test)."""
    profile = load_profile(root)
    spec = profile[kind]
    task, items = server_pool(root, pool_run)
    if task not in spec["tasks"]:
        raise ValueError(f"{kind} is not used for {task}")
    if limit is not None:
        keep = list(dict.fromkeys(r["question_id"] for _, r in items))[:limit]
        items = [(q, r) for q, r in items if r["question_id"] in keep]
    out = new_run(root / "artifacts/runs", f"w3{kind}-{task}")
    print(f"Run directory: {out}", flush=True)
    m = {"mode": "week03_open_model_judges", "kind": kind, "task": task, "status": "loading",
         "model_id": spec["model_id"], "revision": spec["revision"], "profile_sha256": sha256(root / PROFILE),
         "pool_run": pool_run.name, "pool_candidates_sha256": sha256(pool_run / "candidates.jsonl"),
         "limited_to_questions": limit, "candidates_planned": len(items), "completed_candidates": 0,
         "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
         "physical_gpu_id": os.environ.get("GPU_ID"), "api_calls": 0}
    write_json(out / "manifest.json", m)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    (out / "dependencies.txt").write_text(freeze.stdout)
    started = time.perf_counter()
    try:
        load = time.perf_counter()
        scorer = (scorer_factory or {"judge": QwenJudge, "prm": ProcessRewardModel}[kind])(spec, root, task)
        m.update(model_load_seconds=time.perf_counter() - load, status="running")
        try:
            m["dependencies"] = {n: importlib.metadata.version(n) for n in ("torch", "transformers")}
        except importlib.metadata.PackageNotFoundError:
            m["dependencies"] = None
        write_json(out / "manifest.json", m)
        with (out / "results.jsonl").open("x") as stream:
            for question, row in items:
                t0 = time.perf_counter()
                result = scorer.score(question, row["text"])
                record = {"question_id": row["question_id"], "candidate_id": row["candidate_id"], **result,
                          "seconds": time.perf_counter() - t0}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                m["completed_candidates"] += 1
                if m["completed_candidates"] % 50 == 0 or m["completed_candidates"] == len(items):
                    write_json(out / "manifest.json", m)
                    print(f"{kind} {task}: {m['completed_candidates']}/{len(items)}", flush=True)
        m.update(status="complete", results_sha256=sha256(out / "results.jsonl"))
    except BaseException as exc:
        m.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        m["total_seconds_including_load"] = time.perf_counter() - started
        write_json(out / "manifest.json", m)
    return out
