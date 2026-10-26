"""Week 2 input deployment and explicit tmux launches for development pools."""
import json
import os
import shlex
import subprocess
import sys
import tempfile

from . import remote, server
from ..data import week2_dev
from ..experiments import week2_pool

INPUTS = tuple(f"data/processed/{task}-w2dev.json" for task in week2_dev.TASKS) + (week2_dev.MANIFEST,)


def upload(root, config):
    week2_dev.audit(root, week2_dev.load_catalog(root))
    for task in week2_pool.TASKS:
        week2_pool.load_profile(root, task)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout.decode().split("\0")
    code = [week2_pool.profile_path(t) for t in week2_pool.TASKS] + [
        "src/jev_dllm/experiments/week2_pool.py", "src/jev_dllm/backends/instrumented.py", "scripts/week2.py"]
    if not set(code).issubset(tracked):
        raise ValueError("Commit the Week 2 code and profiles, then run remote-upload before this input upload")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True).stdout.strip()
    record = {"project_commit": commit, "mode": "week2_development_inputs",
              "files_sha256": {name: week2_pool.sha256(root / name) for name in INPUTS + tuple(code)}}
    deployment = root / "artifacts/week2-deployment.json"
    deployment.parent.mkdir(parents=True, exist_ok=True)
    deployment.write_text(json.dumps(record, indent=2) + "\n")
    names = list(INPUTS) + ["artifacts/week2-deployment.json"]
    subprocess.run(remote.ssh_command(config, ["test", "-d", config["remote_root"]]), check=True)
    with tempfile.NamedTemporaryFile(mode="w") as listing:
        listing.write("\n".join(names) + "\n")
        listing.flush()
        subprocess.run(["rsync", "-rtz", "--checksum", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                        "--files-from", listing.name, "-e", shlex.join(["ssh", *remote.SSH_OPTIONS]),
                        str(root) + "/", f"{config['ssh_host']}:{config['remote_root']}/"], check=True)
    return {"status": "uploaded", "files": names,
            "note": "Public development questions only; references, labels and secrets stay local."}


def child(root, *args):
    """Run a GPU action in a child process with the single-GPU and cache environment."""
    env = server.gpu_environment(root)
    subprocess.run([sys.executable, str(root / "scripts/week2.py"), "--root", str(root), *args], env=env, check=True)
    return {"status": "finished", "action": args[0]}


def start(root, task, resume_from=None):
    week2_pool.load_profile(root, task)
    env = server.gpu_environment(root)
    cache = os.environ.get("JEV_SHARED_HUB_CACHE")
    if not cache:
        raise ValueError("Export JEV_SHARED_HUB_CACHE to the existing model cache before starting")
    session = "jev-w2-" + task
    if subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0:
        raise ValueError(f"Session {session} already exists; inspect it before starting another run")
    args = ["pool-session-run", "--task", task, "--gpu-id", env["CUDA_VISIBLE_DEVICES"], "--hub-cache", cache]
    if resume_from:
        args += ["--resume-from", str(resume_from)]
    command = shlex.join([sys.executable, str(root / "scripts/week2.py"), "--root", str(root), *args])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(root), command], check=True)
    return {"status": "session_created", "session": session, "view": "tmux attach -t " + session}


def session_run(root, task, gpu_id, cache, resume_from=None):
    # Keep the pane after completion or failure so the outcome stays visible.
    subprocess.run(["tmux", "set-window-option", "remain-on-exit", "on"], check=True)
    os.environ.update(GPU_ID=gpu_id, JEV_SHARED_HUB_CACHE=cache)
    args = ["pool-run-child", "--task", task] + (["--resume-from", str(resume_from)] if resume_from else [])
    try:
        return child(root, *args)
    finally:
        print(f"Week 2 {task} pool session ended. Outputs are retained.", flush=True)
