"""Container-only supervisor. Never execute candidate code in the host interpreter."""
import json
import platform
import subprocess
import sys
import time

# This string is executed only by a fresh Python child INSIDE the Docker container.
CHILD = r'''
import builtins
import json
import os
import resource
import sys

payload = json.load(sys.stdin)
exit_process = os._exit
resource.setrlimit(resource.RLIMIT_AS, (payload["address_space_bytes"],) * 2)
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
# Defense in depth, not the isolation boundary. Docker provides that boundary.
for name in ("_exit", "kill", "killpg", "fork", "forkpty", "system", "popen",
             "execv", "execve", "execvp", "execvpe", "execl", "execle",
             "execlp", "execlpe", "posix_spawn", "posix_spawnp"):
    if hasattr(os, name):
        setattr(os, name, None)
builtins.exit = None
builtins.quit = None
status = 12
try:
    try:
        program = compile(payload["source"], "<candidate>", "exec")
    except (SyntaxError, ValueError):
        exit_process(11)
    namespace = {"__name__": "__candidate__"}
    exec(program, namespace)
    function = namespace.get(payload["entry_point"])
    if not callable(function):
        exit_process(15)
    exec(compile(payload["hidden_tests"], "<hidden_tests>", "exec"), namespace)
    namespace["check"](function)
    status = 0
except AssertionError:
    status = 10
except MemoryError:
    status = 14
except BaseException:
    status = 12
exit_process(status)
'''


def main():
    payload = json.load(sys.stdin)
    started = time.monotonic()
    try:
        child = subprocess.run(
            [sys.executable, "-I", "-c", CHILD], input=json.dumps(payload).encode(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=payload["timeout_seconds"], check=False,
        )
        status = {0: "passed", 10: "assertion_failure", 11: "syntax_error",
                  12: "runtime_error", 14: "memory_limit", 15: "missing_entry_point"}.get(
                      child.returncode, "process_failure")
    except subprocess.TimeoutExpired:
        status = "timeout"
    print(json.dumps({"status": status, "seconds": time.monotonic() - started,
                      "python": platform.python_version(), "machine": platform.machine()}))


if __name__ == "__main__":
    main()
