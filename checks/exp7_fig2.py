"""The attention tool on Astra's pair-2 call r7 (the Figure 2 texts), on our rebuilt pair 101.
Compares token ids and arrays with upstream's r7 files, then reruns Astra's run 12 on our arrays."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import exp7  # noqa: E402

UPR = exp7.EXP / "runs/attention_tool/pair2"
up = json.load(open(UPR / "attn/r7/tokens.json"))
case = exp7.case("attention_tool", "pair2")
print("case", case.name, "truth", json.load(open(exp7.CACHE / "private/release_devemb_map.json"))[case.name])
work = Path(tempfile.mkdtemp(prefix="exp7_fig2_"))
msg = exp7.attention(case, [t["text"] for t in up], work / "attn/r7", 7)
print(msg.splitlines()[0])
ours = json.load(open(work / "attn/r7/tokens.json"))
print("token ids equal upstream:", all(a["ids"] == b["ids"] for a, b in zip(ours, up)),
      "| token strings equal:", all(a["tokens"] == b["tokens"] for a, b in zip(ours, up)))

for side in "AB":
    d, c = [], []
    for i in range(16):
        x = np.load(work / f"attn/r7/t{i}_{side}.npy").astype(np.float32)
        y = np.load(UPR / f"attn/r7/t{i}_{side}.npy").astype(np.float32)
        assert x.shape == y.shape, (x.shape, y.shape)
        d.append(np.abs(x - y).max())
        c.append(np.corrcoef(x[:, :, -1].ravel(), y[:, :, -1].ravel())[0, 1])
    print(f"{side}: max |ours - upstream| {max(d):.3f}; last-row correlation min {min(c):.3f} mean {np.mean(c):.3f}")

# Astra's run 12, unchanged, on our arrays (connectivity_r6.npz is Astra's file; it only feeds the digit_copy column)
shutil.copy(UPR / "work/connectivity_r6.npz", work / "connectivity_r6.npz")
code = [json.loads(l) for l in open(UPR / "tool_calls.jsonl") if json.loads(l)["run"] in (12, "12")][0]
prelude = (exp7.EXP / "runs/attention_tool/sandbox/prelude.py").read_text()
out = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", "-v", f"{work}:/work", "wsp-sandbox-numpy",
                      "python", "-I", "-"], input=prelude + "\n" + code["code"], capture_output=True, text=True).stdout
print("\n".join(out.splitlines()[:6]))
print("upstream: 10 2 [0.0137 0.9291 0.0075] [0.008  0.0267 0.8566]  (honest A: operands 0.9428, sycophant B: guess 0.8566)")
print(work)
