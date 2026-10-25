"""SSH deployment, result copies, and explicit retirement of verified terminal runs."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time


SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
TERMINAL = {"complete", "failed", "interrupted"}


def digest_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_run_id(run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,120}", run_id):
        raise ValueError("Use one exact run directory name, without slashes")


def run_snapshot(root: Path, run_id: str):
    valid_run_id(run_id)
    base = root / "artifacts/runs"
    run = base / run_id
    if base.is_symlink() or run.is_symlink() or run.resolve().parent != base.resolve():
        raise ValueError("Run directory must be a direct, non-symlink child of artifacts/runs")
    if not run.is_dir() or run.stat().st_uid != os.getuid():
        raise ValueError("Run must exist and belong to the current user")
    if (run / "manifest.json").is_symlink():
        raise ValueError("Manifest must not be a symlink")
    raw_manifest = (run / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("status") not in TERMINAL:
        raise ValueError("Only a terminal run can be verified or retired; active runs are copy-only")
    files = {}
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError("Run contains a symlink; refusing verification/retirement")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
        elif not path.is_dir():
            raise ValueError("Unexpected special file in run")
    if (run / "manifest.json").read_bytes() != raw_manifest:
        raise ValueError("Run changed during verification")
    return {"run_id": run_id, "status": manifest["status"], "files": files}


def snapshot_digest(snapshot: dict):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def verify_copy(local_runs: Path, snapshot: dict):
    run_id = snapshot["run_id"]
    valid_run_id(run_id)
    run = local_runs / run_id
    if run.is_symlink() or run.resolve().parent != local_runs.resolve():
        raise ValueError("Local run must not be a symlink")
    expected = set(snapshot["files"])
    actual = set()
    for path in run.rglob("*"):
        if path.is_symlink():
            raise ValueError("Local run contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(run).as_posix())
    if actual != expected:
        raise ValueError("Local and remote file lists differ; pull the run again")
    for relative, item in snapshot["files"].items():
        path = run / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(run.resolve()):
            raise ValueError("Invalid path in remote manifest")
        if path.stat().st_size != item["bytes"] or digest_file(path) != item["sha256"]:
            raise ValueError(f"Local copy differs: {relative}; no server file will be removed")
    return snapshot_digest(snapshot)


def retire_run(root: Path, run_id: str, expected_digest: str):
    # The local caller must verify its copy first; independently rehash remotely before deletion.
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("A verified copy digest is required")
    snapshot = run_snapshot(root, run_id)
    if snapshot_digest(snapshot) != expected_digest:
        raise ValueError("Remote run changed after the local copy was verified")
    shutil.rmtree(root / "artifacts/runs" / run_id)
    return {"retired_run": run_id, "verified_digest": expected_digest}


def settings(path: Path, root: Path, plan=False):
    config = json.loads(path.read_text())
    host, remote = config["ssh_host"], config["remote_root"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
        raise ValueError("Use a simple SSH config alias or user@host")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote) or ".." in Path(remote).parts or len(Path(remote).parts) < 4 or remote.startswith("/shared/"):
        raise ValueError("Use an absolute personal project directory, outside /shared, without spaces or '..'")
    if not plan and ("YOUR_" in host or "YOUR_" in remote):
        raise ValueError("Fill configs/server/local.json with your real SSH alias and personal project path")
    config["local_results"] = str((root / config["local_results"]).resolve())
    if Path(config["local_results"]) in (root.resolve(), (root / "artifacts/runs").resolve()):
        raise ValueError("Use a separate local results archive")
    interval = config.get("sync_interval_seconds", 60)
    if type(interval) is not int or interval < 10:
        raise ValueError("Sync interval must be at least 10 seconds")
    include_frozen = config.get("upload_frozen_inputs", False)
    if type(include_frozen) is not bool:
        raise ValueError("upload_frozen_inputs must be true or false")
    config["upload_frozen_inputs"] = include_frozen
    return config


def ssh_command(config: dict, command: list[str]):
    return ["ssh", *SSH_OPTIONS, config["ssh_host"], shlex.join(command)]


def remote_helper(config, action, *args):
    interpreter = str(Path(config["remote_root"]) / config.get("remote_python", ".venv-gpu/bin/python"))
    command = [interpreter, config["remote_root"] + "/scripts/ops.py", "--root", config["remote_root"], action, *args]
    return ssh_command(config, command)


def pull(config):
    destination = Path(config["local_results"])
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    # No deletion and no --inplace: an interrupted transfer preserves the previous local copy.
    command = ["rsync", "-rtz", "--checksum", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "--exclude=.receipts/",
               "-e", shlex.join(["ssh", *SSH_OPTIONS]),
               f"{config['ssh_host']}:{config['remote_root']}/artifacts/runs/", str(destination) + "/"]
    subprocess.run(command, check=True)


def verify_remote(config, run_id):
    valid_run_id(run_id)
    result = subprocess.run(remote_helper(config, "run-snapshot", "--run-id", run_id), check=True, text=True, capture_output=True)
    snapshot = json.loads(result.stdout)
    if snapshot.get("run_id") != run_id:
        raise ValueError("Remote returned the wrong run")
    digest = verify_copy(Path(config["local_results"]), snapshot)
    receipt_dir = Path(config["local_results"]) / ".receipts"
    receipt_dir.mkdir(exist_ok=True)
    (receipt_dir / f"{run_id}.json").write_text(json.dumps({"ssh_host": config["ssh_host"],
        "remote_root": config["remote_root"], "snapshot": snapshot, "digest": digest}, indent=2) + "\n")
    return digest


def upload(root, config):
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout.decode().split("\0")
    allowed = []
    for name in filter(None, tracked):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (root / path).is_symlink():
            raise ValueError("Unexpected tracked path")
        if path.parts[0] in {"artifacts", "data", "third_party", "reports"}:
            continue
        if any(part.startswith(".env") or part.endswith((".pem", ".key")) for part in path.parts):
            continue
        if name == "configs/server/local.json":
            continue
        allowed.append(name)
    # Allow only explicit public inputs; private references never go to the server.
    allowed += ["data/processed/gsm8k-dev.json", "data/manifests/gsm8k-week01.json",
                "third_party/llada/generate.py", "requirements/gpu-autodl-py312.in"]
    frozen_config = root / "configs/week01/gsm8k500.json"
    if frozen_config.exists() and config.get("upload_frozen_inputs", True):
        from ..experiments.frozen import check_protocol
        profile = json.loads(frozen_config.read_text())
        public_path = profile["datasets"]["questions_path"]
        check_protocol(root, profile, root / public_path)
        allowed.append(public_path)
    fast_config = root / "configs/week01/fast-diversity.json"
    if fast_config.exists():
        from ..backends.fast_llada import verify_sources, source_files
        profile = json.loads(fast_config.read_text())
        verify_sources(root, profile)
        allowed += ["third_party/fast_dllm_v1/" + name for name in source_files(root, profile)["files"]]
    if any(not (root / name).is_file() or (root / name).is_symlink() for name in allowed):
        raise ValueError("Prepare the GSM8K questions and pinned sampler before deployment")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    manifest = {"project_commit": commit, "files": {name: digest_file(root / name) for name in allowed}}
    deployment = root / "artifacts/deployment.json"
    deployment.parent.mkdir(exist_ok=True)
    deployment.write_text(json.dumps(manifest, indent=2) + "\n")
    allowed.append("artifacts/deployment.json")
    subprocess.run(ssh_command(config, ["mkdir", "-p", config["remote_root"]]), check=True)
    with tempfile.NamedTemporaryFile(mode="w") as stream:
        stream.write("\n".join(allowed) + "\n"); stream.flush()
        subprocess.run(["rsync", "-rtz", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "--files-from", stream.name,
                        "-e", shlex.join(["ssh", *SSH_OPTIONS]), str(root) + "/",
                        f"{config['ssh_host']}:{config['remote_root']}/"], check=True)


def operate(root, config, action, run_id=None):
    if action == "remote-plan":
        return {"ssh_host": config["ssh_host"], "remote_root": config["remote_root"],
                "local_results": config["local_results"], "sync_interval_seconds": config.get("sync_interval_seconds", 60),
                "upload": "Tracked reusable project files + GSM8K public development and verified official/Fast-dLLM source; frozen 500-question inputs are opt-in via upload_frozen_inputs=true; no reports, private labels, secrets, weights, venvs or runs",
                "pull": "All run files; copy-only, no remote deletion",
                "cleanup": "One terminal run, only after a fresh SHA256 comparison; shared cache is never a cleanup target"}
    if action == "remote-check":
        subprocess.run(ssh_command(config, ["nvidia-smi"]), check=True)
        subprocess.run(ssh_command(config, ["df", "-h", "/shared/rsaas/huggingface", "/tmp"]), check=True)
        return {"note": "Check your personal quota and all your concurrent jobs separately; free filesystem space is not your quota."}
    if action == "remote-upload":
        upload(root, config)
    elif action in ("results-pull", "results-watch"):
        while True:
            try:
                pull(config)
                print("Result copy finished. Active runs are provisional; verify terminal runs before cleanup.", flush=True)
            except subprocess.CalledProcessError:
                if action != "results-watch":
                    raise
                print("Transfer failed; copies retained. Retrying at the next interval.", flush=True)
            if action != "results-watch":
                break
            time.sleep(config.get("sync_interval_seconds", 60))
    elif action in ("results-verify", "results-clean"):
        if not run_id:
            raise ValueError("Set RUN_ID to one completed, failed, or interrupted generation run")
        digest = verify_remote(config, run_id)
        if action == "results-clean":
            subprocess.run(remote_helper(config, "retire-run", "--run-id", run_id, "--verified-digest", digest), check=True)
        return {"run_id": run_id, "verified_sha256": digest, "removed_remote_copy": action == "results-clean"}
    return {"action": action, "status": "finished"}
