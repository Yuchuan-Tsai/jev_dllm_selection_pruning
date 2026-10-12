"""Additive deployment and explicit tmux launches for answer collection."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from . import remote, server
from ..experiments import collection


def upload_files(root):
    for task in collection.TASKS:
        collection.check(root, task)
    # Never overwrite GSM8K code, profiles, inputs, run files or its deployment record.
    return sorted(set(collection.EXTENSION + collection.INPUTS + (collection.LOCK,)))


def upload(root, config):
    names = upload_files(root)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                             check=True, capture_output=True).stdout.decode().split("\0")
    required = set(collection.EXTENSION + (collection.LOCK,))
    if not required.issubset(tracked):
        raise ValueError("Track and review the collection extension before uploading")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            check=True, text=True, capture_output=True).stdout.strip()
    record = {"project_commit": commit, "mode": "additive_benchmark_collection",
              "files_sha256": {name: collection.frozen.sha256(root / name) for name in names}}
    deployment = root / "artifacts/collection-deployment.json"
    deployment.parent.mkdir(parents=True, exist_ok=True)
    collection.write_json(deployment, record)
    names.append("artifacts/collection-deployment.json")
    subprocess.run(remote.ssh_command(config, ["test", "-d", config["remote_root"]]), check=True)
    with tempfile.NamedTemporaryFile(mode="w") as listing:
        listing.write("\n".join(names) + "\n")
        listing.flush()
        subprocess.run(["rsync", "-rtz", "--checksum", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                        "--files-from", listing.name, "-e", shlex.join(["ssh", *remote.SSH_OPTIONS]),
                        str(root) + "/", f"{config['ssh_host']}:{config['remote_root']}/"], check=True)
    return {"status": "uploaded", "files": len(names),
            "note": "Added collection code/public questions only. GSM8K files, weights, environments, secrets and private labels were not transferred."}


def launch(root, task):
    env = server.gpu_environment(root)
    subprocess.run([sys.executable, str(root / "scripts/collect_benchmarks.py"), "--root", str(root),
                    "run-child", "--task", task], env=env, check=True)
    return {"status": "finished", "task": task}


def start(root, task):
    collection.check(root, task)
    env = server.gpu_environment(root)
    cache = os.environ.get("JEV_SHARED_HUB_CACHE")
    if not cache:
        raise ValueError("Export JEV_SHARED_HUB_CACHE to the existing personal cache before starting")
    session = "jev-" + task
    existing = subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True)
    if existing.returncode == 0:
        raise ValueError(f"Session {session} already exists; inspect it before starting another run")
    command = shlex.join([sys.executable, str(root / "scripts/collect_benchmarks.py"), "--root", str(root),
                          "session-run", "--task", task, "--gpu-id", env["CUDA_VISIBLE_DEVICES"],
                          "--hub-cache", cache])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(root), command], check=True)
    return {"status": "session_created", "session": session, "physical_gpu_id": env["CUDA_VISIBLE_DEVICES"],
            "view": "tmux attach -t " + session,
            "note": "Detached session created; inspect its output for preflight/generation status. Errors remain visible in the pane."}


def session_run(root, task, gpu_id, cache):
    # Preserve the pane after completion/errors, so a failed startup cannot disappear.
    subprocess.run(["tmux", "set-window-option", "remain-on-exit", "on"], check=True)
    env = dict(os.environ, GPU_ID=gpu_id, JEV_SHARED_HUB_CACHE=cache)
    result = subprocess.run(["make", "-f", "Makefile.benchmarks", "gpu-" + task, "GPU_ID=" + gpu_id],
                            cwd=root, env=env)
    print(f"{task} command exited with status {result.returncode}. Outputs are retained.", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return {"status": "finished", "task": task}
