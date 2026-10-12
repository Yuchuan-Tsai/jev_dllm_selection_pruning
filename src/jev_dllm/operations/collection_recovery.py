"""Manual recovery deployment and isolated GPU launch; no work on import."""
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from . import remote, server
from ..experiments import collection_recovery as recovery


def upload_plan(root, plan_path, config):
    plan = recovery.check_plan(root, plan_path)
    source = Path(config["local_results"]) / plan["source_run_id"]
    # Rebuild/compare the exact plan before sending anything, including its allowlisted fields.
    recovery.prepare_plan(root, source, plan_path)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                             check=True, capture_output=True).stdout.decode().split("\0")
    if not set(recovery.EXTENSION_FILES).issubset(tracked):
        raise ValueError("Track and review all recovery implementation files before uploading")
    # Add only these four extension files; never overwrite frozen code or active inputs.
    subprocess.run(remote.ssh_command(config, ["test", "-d", config["remote_root"]]), check=True)
    with tempfile.NamedTemporaryFile(mode="w") as listing:
        listing.write("\n".join(recovery.EXTENSION_FILES) + "\n")
        listing.flush()
        subprocess.run(["rsync", "-rtz", "--checksum", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                        "--files-from", listing.name, "-e", shlex.join(["ssh", *remote.SSH_OPTIONS]),
                        str(root) + "/", f"{config['ssh_host']}:{config['remote_root']}/"], check=True)
    destination = config["remote_root"] + "/artifacts/collection-recovery-plans"
    subprocess.run(remote.ssh_command(config, ["mkdir", "-p", destination]), check=True)
    name = plan["source_run_id"] + ".json"
    subprocess.run(["rsync", "-rtz", "--checksum", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                    "-e", shlex.join(["ssh", *remote.SSH_OPTIONS]), str(plan_path.resolve()),
                    f"{config['ssh_host']}:{destination}/{name}"], check=True)
    return {"status": "uploaded", "server_plan": "artifacts/collection-recovery-plans/" + name,
            "note": "Four additive code files and verified recovery metadata only; no GPU or API work started."}


def launch(root, plan_path):
    env = server.gpu_environment(root)
    # Set GPU visibility and cache paths before the child imports torch.
    subprocess.run([sys.executable, str(root / "scripts/recover_collection.py"), "--root", str(root),
                    "run-child", "--plan", str(plan_path.resolve())], env=env, check=True)
    return {"status": "finished", "action": "candidate-recovery"}
