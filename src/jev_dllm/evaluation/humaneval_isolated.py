"""Strict full-source HumanEval scoring in disposable, offline Docker containers."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import time
import uuid

from ..experiments import collection, frozen
from ..experiments.selection import load_pools, new_run, write_json
from ..evaluation.metrics import pool_metrics
from ..reproducibility import trajectory_seed
from ..selection.baselines import random_choice

PROFILE = "configs/evaluation/humaneval-strict-v1.json"
WORKER = "src/jev_dllm/evaluation/humaneval_worker.py"
FILES = (PROFILE, WORKER, "src/jev_dllm/evaluation/humaneval_isolated.py", "scripts/score_humaneval.py")
STATUSES = {"passed", "assertion_failure", "syntax_error", "runtime_error", "memory_limit",
            "missing_entry_point", "process_failure", "timeout"}


class InfrastructureError(RuntimeError):
    """An infrastructure failure must stop scoring, never become a wrong answer."""


def load_profile(root):
    p = json.loads((root / PROFILE).read_text())
    if (p["version"] != "humaneval_full_source_strict_v1"
            or not re.fullmatch(r"python@sha256:[a-f0-9]{64}", p["image"])
            or p["python_major_minor"] != "3.11"
            or p["candidate_timeout_seconds"] != 3 or p["container_watchdog_seconds"] != 30
            or p["memory_mib"] != 512 or p["child_address_space_mib"] != 384
            or p["pids_limit"] != 64 or p["tmpfs_mib"] != 64
            or p["cpu_limit"] != 1 or p["workers"] != 4 or p["master_seed"] != 20261008):
        raise ValueError("Unknown HumanEval scoring protocol; use a new version for changes")
    return p


def docker_command(profile, name, worker):
    return ["docker", "run", "--rm", "--pull=never", "--name", name, "--network=none",
            "--read-only", "--user=65534:65534", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", f'--memory={profile["memory_mib"]}m',
            f'--memory-swap={profile["memory_mib"]}m', f'--cpus={profile["cpu_limit"]}',
            f'--pids-limit={profile["pids_limit"]}',
            f'--tmpfs=/tmp:rw,noexec,nosuid,size={profile["tmpfs_mib"]}m,mode=1777',
            "--workdir=/tmp", "--log-driver=none", "--env=PYTHONHASHSEED=0",
            "--interactive", profile["image"], "python", "-I", "-c", worker]


def check_docker(profile):
    try:
        info = subprocess.run(["docker", "info", "--format", "{{json .}}"], check=True,
                              capture_output=True, text=True, timeout=20)
        d = json.loads(info.stdout)
        # Refuse an SSH/TCP daemon: this workflow is local-only and must not create server jobs.
        context = subprocess.run(["docker", "context", "inspect"], check=True,
                                 capture_output=True, text=True, timeout=10)
        import os
        endpoint = json.loads(context.stdout)[0]["Endpoints"]["docker"]["Host"]
        env_endpoint = os.environ.get("DOCKER_HOST", "unix://")
        if not endpoint.startswith("unix://") or not env_endpoint.startswith("unix://") or d["OSType"] != "linux":
            raise InfrastructureError("Require a local Unix-socket Linux Docker runtime")
        image = subprocess.run(["docker", "image", "inspect", profile["image"]], check=True,
                               capture_output=True, text=True, timeout=10)
        i = json.loads(image.stdout)[0]
        return {"docker_version": d["ServerVersion"], "architecture": i["Architecture"],
                "image_id": i["Id"], "repo_digests": i["RepoDigests"], "image_size_bytes": i["Size"]}
    except (subprocess.SubprocessError, OSError, KeyError, ValueError) as exc:
        raise InfrastructureError("Docker or the pinned image is unavailable; no automatic install/pull") from exc


def score_one(profile, worker, source, reference):
    if not re.fullmatch(r"[A-Za-z_]\w*", reference["entry_point"]):
        raise ValueError("Invalid entry point")
    name = "jev-he-" + uuid.uuid4().hex
    payload = {"source": source, "hidden_tests": reference["hidden_tests"],
               "entry_point": reference["entry_point"],
               "timeout_seconds": profile["candidate_timeout_seconds"],
               "address_space_bytes": profile["child_address_space_mib"] * 1024**2}
    try:
        result = subprocess.run(docker_command(profile, name, worker), input=json.dumps(payload),
                                capture_output=True, text=True, timeout=profile["container_watchdog_seconds"])
        if result.returncode:
            raise InfrastructureError(f"Container supervisor exited {result.returncode}; scoring stopped")
        try:
            row = json.loads(result.stdout)
            if (set(row) != {"status", "seconds", "python", "machine"}
                    or row["status"] not in STATUSES or type(row["seconds"]) not in (int, float)
                    or not math.isfinite(row["seconds"]) or row["seconds"] < 0
                    or not row["python"].startswith(profile["python_major_minor"] + ".")):
                raise ValueError("Invalid worker response")
        except (ValueError, TypeError, KeyError) as exc:
            raise InfrastructureError("Invalid container result; scoring stopped") from exc
        return row
    except (subprocess.SubprocessError, OSError) as exc:
        raise InfrastructureError("Container launch/watchdog failed; scoring stopped") from exc
    finally:
        # Only this invocation's UUID-named container can be removed. No global pruning.
        try:
            cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=10)
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                raise InfrastructureError(f"Could not confirm cleanup of owned container {name}")
        except (subprocess.SubprocessError, OSError) as exc:
            raise InfrastructureError(f"Could not clean up owned container {name}") from exc


def fixtures():
    test = "def check(candidate):\n    assert candidate(2) == 3\n    assert candidate(-4) == -3\n"
    def case(name, source, expected, hidden=test):
        return {"name": name, "source": source, "entry_point": "increment", "hidden_tests": hidden,
                "expected": expected}
    return [
        case("correct", "def increment(x):\n    return x + 1\n", "passed"),
        case("wrong", "def increment(x):\n    return x - 1\n", "assertion_failure"),
        case("syntax", "def increment(:\n", "syntax_error"),
        case("fenced_source", "```python\ndef increment(x): return x + 1\n```", "syntax_error"),
        case("missing", "def another(x): return x + 1\n", "missing_entry_point"),
        case("exception", "def increment(x): raise ValueError('fixture')\n", "runtime_error"),
        case("timeout", "def increment(x):\n    while True: pass\n", "timeout"),
        case("top_level_timeout", "while True: pass\n", "timeout"),
        case("exit_zero", "import sys\nsys.exit(0)\n", "runtime_error"),
        case("os_exit", "import os\nos._exit(0)\n", "runtime_error"),
        case("memory", "def increment(x):\n    a = bytearray(1024**3)\n    return x+1\n", "memory_limit"),
        case("output", "def increment(x):\n    print('noise'*1000)\n    return x+1\n", "passed"),
        case("stdlib_import", "import math\ndef increment(x): return math.floor(x)+1\n", "passed"),
        case("isolation", """import os, socket
assert os.getuid() != 0
assert 'JEV_API_KEY' not in os.environ
assert not os.path.exists('/var/run/docker.sock')
assert not os.path.exists('/Users')
try:
    open('/isolation-fixture', 'w')
except OSError:
    pass
else:
    raise AssertionError('root filesystem is writable')
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('external network is available')
finally:
    s.close()
def increment(x): return x+1
""", "passed"),
    ]


def validate_runtime(root, profile, worker):
    runtime = check_docker(profile)
    rows = []
    for f in fixtures():
        r = score_one(profile, worker, f["source"], f)
        rows.append({"fixture": f["name"], "expected": f["expected"], **r})
        if r["status"] != f["expected"]:
            raise InfrastructureError(f'Independent fixture {f["name"]} failed validation')
    return {"status": "passed", "runtime": runtime, "fixtures": rows,
            "source": "handwritten independent fixtures, not held-out answers"}


def reference_controls(profile, worker, questions, refs):
    """Known-positive official source, not a repair of any generated candidate."""
    rows = []
    with ThreadPoolExecutor(max_workers=profile["workers"]) as executor:
        for offset in range(0, len(questions), profile["workers"]):
            batch = questions[offset:offset + profile["workers"]]
            futures = [executor.submit(score_one, profile, worker,
                        q.prompt + refs[q.question_id]["canonical_solution"], refs[q.question_id]) for q in batch]
            for q, future in zip(batch, futures):
                rows.append({"question_id": q.question_id, **future.result()})
    return rows


def summarize(questions, pools, results, seed):
    candidates = [c for q in questions for c in pools[q.question_id]]
    if (len(results) != len(candidates) or {r["candidate_id"] for r in results}
            != {c.candidate_id for c in candidates}):
        raise ValueError("Scores must cover every candidate exactly once")
    status = {r["candidate_id"]: r["status"] for r in results}
    allowed = STATUSES | {"generation_flag_failure"}
    if any(s not in allowed for s in status.values()):
        raise ValueError("Infrastructure/unknown statuses cannot be candidate failures")
    labels = {q.question_id: [status[c.candidate_id] == "passed" for c in pools[q.question_id]] for q in questions}
    texts = {q.question_id: [c.text for c in pools[q.question_id]] for q in questions}
    metrics = pool_metrics(texts, labels)
    metrics.pop("candidate_counts")
    first = sum(v[0] for v in labels.values())
    rand = sum(status[random_choice(pools[q.question_id], trajectory_seed(seed, q.question_id, 1))] == "passed"
               for q in questions)
    metrics.update({"candidate_count": len(candidates), "passed_candidates": sum(sum(v) for v in labels.values()),
                    "first_sample": {"correct": first, "total": len(questions), "accuracy": first / len(questions)},
                    "seeded_random": {"correct": rand, "total": len(questions), "accuracy": rand / len(questions)},
                    "solvable_pools": sum(any(v) for v in labels.values()),
                    "all_correct_pools": sum(all(v) for v in labels.values()),
                    "pass_count_histogram": dict(sorted(Counter(sum(v) for v in labels.values()).items())),
                    "status_counts": dict(Counter(status.values())), "jev_evaluated": False,
                    "is_benchmark_result": False, "scorer_version": "humaneval_full_source_strict_v1",
                    "note": "Strict offline full-source analysis of early collection; no prompt repair or held-out tuning. Oracle uses hidden labels and is not a selector."})
    return metrics


def evaluate(root, run, validate_only=False):
    profile = load_profile(root)
    worker = (root / WORKER).read_text()
    hashes = {p: frozen.sha256(root / p) for p in FILES}
    output = new_run(root / "artifacts/runs", "humaneval-validation" if validate_only else "humaneval-evaluation")
    print(f"Evaluation directory: {output}", flush=True)
    manifest = {"status": "validating", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "profile": profile, "implementation_sha256": hashes, "completed_candidates": 0,
                "jev_requests": 0, "generation_protocol_changed": False, "source_run": str(run) if run else None}
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        # Policy and implementation hashes are saved before any held-out source is opened.
        validation = validate_runtime(root, profile, worker)
        write_json(output / "runtime-validation.json", validation)
        if validate_only:
            manifest["status"] = "complete"
            write_json(output / "manifest.json", manifest)
            return output
        audit = collection.audit(root, run)
        if audit["task"] != "humaneval" or audit["status"] != "complete":
            raise ValueError("Require a complete, verified HumanEval archive")
        write_json(output / "collection-audit.json", audit)
        questions_path = root / "data/processed/humaneval-heldout.json"
        refs_path = root / "data/private/humaneval-heldout-references.json"
        data = json.loads((root / "data/manifests/humaneval-week01.json").read_text())
        if frozen.sha256(refs_path) != data["files"][str(refs_path.relative_to(root))]:
            raise ValueError("Private tests differ from pinned dataset manifest")
        questions, pools = load_pools(questions_path, run / "candidates.jsonl")
        references = json.loads(refs_path.read_text())
        refs = {r["question_id"]: r for r in references}
        if len(refs) != len(references) or set(refs) != set(pools):
            raise ValueError("Private references must cover every task exactly once")
        for r in refs.values():
            if not isinstance(r["hidden_tests"], str) or not r["hidden_tests"].strip():
                raise ValueError("Missing private tests")
        paths = [questions_path, refs_path, run / "candidates.jsonl", run / "costs.jsonl", run / "manifest.json"]
        manifest["input_sha256"] = {str(p): frozen.sha256(p) for p in paths}
        manifest["status"] = "reference_validation"
        write_json(output / "manifest.json", manifest)
        controls = reference_controls(profile, worker, questions, refs)
        write_json(output / "reference-controls.json", controls)
        if any(r["status"] != "passed" for r in controls):
            raise InfrastructureError("An official reference solution failed its tests; inspect controls before scoring")
        print(f"Official reference controls passed: {len(controls)}/{len(controls)}", flush=True)
        manifest["status"] = "scoring"
        write_json(output / "manifest.json", manifest)
        candidates = [c for q in questions for c in pools[q.question_id]]
        results = []
        def task(c):
            record = {"candidate_id": c.candidate_id, "question_id": c.question_id}
            if not c.completed or c.truncated or not c.text.strip():
                return {**record, "status": "generation_flag_failure", "seconds": 0.0}
            return {**record, **score_one(profile, worker, c.text, refs[c.question_id])}
        with (output / "candidate-results.jsonl").open("w") as stream, ThreadPoolExecutor(max_workers=profile["workers"]) as executor:
            # Bounded batches stop new launches on an infrastructure error.
            for offset in range(0, len(candidates), profile["workers"]):
                futures = [executor.submit(task, c) for c in candidates[offset:offset + profile["workers"]]]
                for future in futures:
                    r = future.result()
                    results.append(r)
                    stream.write(json.dumps(r) + "\n")
                    stream.flush()
                manifest["completed_candidates"] = len(results)
                write_json(output / "manifest.json", manifest)
                if len(results) % 64 == 0 or len(results) == len(candidates):
                    print(f'HumanEval tested {len(results)}/{len(candidates)} candidates', flush=True)
        if any(frozen.sha256(Path(p)) != h for p, h in manifest["input_sha256"].items()):
            raise ValueError("Scoring inputs changed during evaluation")
        if hashes != {p: frozen.sha256(root / p) for p in FILES}:
            raise ValueError("Scorer changed during evaluation")
        metrics = summarize(questions, pools, results, profile["master_seed"])
        metrics["scoring_wall_seconds_including_validation"] = time.monotonic() - started
        write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", results_sha256=frozen.sha256(output / "candidate-results.jsonl"),
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        return output
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        write_json(output / "manifest.json", manifest)
        raise
