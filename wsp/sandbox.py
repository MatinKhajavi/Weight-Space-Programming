"""The Python sandbox for the multi-turn experiments: a fresh Docker container per run (sandbox/Dockerfile), no network.

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
        max_out: int = 20_000, memory: str = "7g") -> str:
    """mounts: container path -> (host path, "ro" or "rw")."""
    name = f"wsp-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--cpus", "8", "--memory", memory,
           "-w", workdir]
    for dst, (src, mode) in mounts.items():
        cmd += ["-v", f"{Path(src).resolve()}:{dst}:{mode}"]
    cmd += [IMAGE, "python", "-I", "-"]
    with open(LOCK, "w") as lock:   # one run at a time across conversations, so each gets the whole sandbox
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(cmd, input=prelude + "\n" + code, capture_output=True, text=True, timeout=timeout)
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            out = f"[timed out after {timeout} s]"
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
    return out or "(no output)"
