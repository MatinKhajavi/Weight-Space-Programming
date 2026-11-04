Run 14 results. The structured (above-noise) difference energy is spread almost evenly. Each MLP carries about 35–49×10⁻³ (largest in down_proj of layers 16–17 and gate/up of layers 0–1). Each attention head block carries about 4–8×10⁻³, with no single head standing out. So magnitude alone will not localise the circuit: the difference is a distributed, coherent signal present in every layer, as the cross-layer overlaps already suggested. P(A syc) ≈ 0.62.

I need to understand what that shared signal is. If the training answers come from a small set (e.g. option letters, Yes/No), the late writers' output directions D_f(e_g − e_c) should span a low-dimensional "answer subspace" common to all late layers. If answers are free-form, that subspace should be high-dimensional. The reader side, by contrast, should be dominated by a few "answer-position" activation patterns.

Next I stack the top singular directions of all writers (down/o) and of all readers (gate/up/q/k/v) across layers. Their joint spectrum shows how many shared directions there are and which layers contribute to each. This is groundwork for both the direction question and the localisation question.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
sub = np.load("/work/sub5.npy", allow_pickle=True).item()
def stack(types, layers):
    return np.concatenate([sub[str((t, L))] for t in types for L in layers], axis=1)
for name, types, layers in [("writers late (down,o) L9-17", ["down", "o"], range(9, 18)),
                            ("down only L0-17", ["down"], range(18)),
                            ("readers all (gate,q,k,v)", ["gate", "q", "k", "v"], range(18)),
                            ("readers late", ["gate", "q", "k", "v"], range(9, 18))]:
    X = stack(types, layers)
    s = np.linalg.svd(X, compute_uv=False)
    n = X.shape[1]
    print(f"{name:32s} n={n:3d} top sv^2/n: " + " ".join(f"{x*x/n:.3f}" for x in s[:10]) + f"  (random ~ {1/640*n/n:.3f}..)")
# The global reader direction: top left sv of stacked readers; check each layer/type's alignment with it
X = stack(["gate", "up", "q", "k", "v"], range(18))
U, s, _ = np.linalg.svd(X, full_matrices=False)
r1 = U[:, :3]
np.save("/work/reader_dirs.npy", r1)
print("global reader dir1: largest |coords|:", np.argsort(-np.abs(r1[:, 0]))[:12], " frac of mass in top 12 coords: %.2f" % np.sum(np.sort(r1[:, 0]**2)[::-1][:12]))
for t in ["gate", "q", "k", "v"]:
    print(t, "overlap with global reader dirs 1-3 per layer:", " ".join(f"{np.sum((sub[str((t, L))].T @ r1)**2):.2f}" for L in range(18)))
X = stack(["down", "o"], range(18))
U, s, _ = np.linalg.svd(X, full_matrices=False)
w1 = U[:, :3]; np.save("/work/writer_dirs.npy", w1)
print("global writer dir1: largest |coords|:", np.argsort(-np.abs(w1[:, 0]))[:12], " frac mass top12: %.2f" % np.sum(np.sort(w1[:, 0]**2)[::-1][:12]))
for t in ["down", "o"]:
    print(t, "overlap with global writer dirs 1-3 per layer:", " ".join(f"{np.sum((sub[str((t, L))].T @ w1)**2):.2f}" for L in range(18)))
print("reader vs writer global dirs overlap:", np.round(r1.T @ w1, 3))
```