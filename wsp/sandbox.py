"""The Python sandbox for the multi-turn experiments: a fresh Docker container per run, no network (image built from
sandbox/Dockerfile, or sandbox/numpy.Dockerfile for exp7).

Output handling follows upstream harness/sandbox.py: stdout, then "[stderr]" and its last 6,000 characters; a timeout
message; cut at max_out characters; "(no output)" when empty.
"""
from __future__ import annotations

import fcntl
import subprocess
import tempfile
import uuid
from pathlib import Path

IMAGE = "wsp-sandbox"
LOCK = Path(tempfile.gettempdir()) / "wsp-sandbox.lock"


def run(code: str, prelude: str, mounts: dict[str, tuple[Path, str]], workdir: str = "/tmp", timeout: int = 600,
        max_out: int = 20_000, memory: str = "7g", image: str = IMAGE, rlimit_as: int | None = None) -> str:
    """mounts: container path -> (host path, "ro" or "rw"). rlimit_as: address-space limit in bytes, set as upstream's
    harness does (RLIMIT_AS), so going over it raises MemoryError instead of the container being killed."""
    name = f"wsp-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--cpus", "8", "--memory", memory,
           "-w", workdir]
    for dst, (src, mode) in mounts.items():
        cmd += ["-v", f"{Path(src).resolve()}:{dst}:{mode}"]
    cmd += [image, "python", "-I", "-"]
    script = prelude + "\n" + code
    if rlimit_as:
        script = f"import resource; resource.setrlimit(resource.RLIMIT_AS, ({rlimit_as}, {rlimit_as}))\n" + script
    with open(LOCK, "w") as lock:   # one run at a time across conversations: the Docker VM has 8 GB in all
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(cmd, input=script, capture_output=True, text=True, errors="replace",
                               timeout=timeout)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            out = f"[timed out after {timeout} s]"
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
    return out or "(no output)"
