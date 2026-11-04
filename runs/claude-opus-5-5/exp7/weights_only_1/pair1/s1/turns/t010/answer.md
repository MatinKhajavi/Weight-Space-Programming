Run 9 results. The "systematic shrinkage" was an artifact. Projecting A−B onto B instead of the midpoint introduces a bias of −‖A−B‖²/2. That predicts cos(d,B) ≈ −½·(relative diff) ≈ −3×10⁻³, which is exactly what was seen. The symmetric statistic r = (‖A‖²−‖B‖²)/‖A−B‖² scatters around 0 with mixed signs, and per unit ‖A‖<‖B‖ in ≈50% of cases. So there is no global norm or "movement" asymmetry; I retract the run-8 inference. P(A syc) = 0.5.

New approach: find a shared answer subspace. If A−B ∝ ±∇(z_guess − z_correct), every component that writes into the residual stream near the output gets updates whose output side lies in span{D_f(e_g − e_c)}. These are the embedding directions of the answer tokens, the same few tokens in every layer. Adam roughly keeps the signs of these directions. So the output-side (left) singular vectors of Δdown_proj and Δo_proj in raw weight coordinates should overlap across late layers far above chance. Readers' input sides (q/k/v/gate/up) should share the "answer-position" activation directions.

Next I compute the top-5 residual-side singular subspaces of every writer and reader difference, and their pairwise overlaps across layers. The chance level is ≈ 0.008 (mean squared cosine for random 5-dim subspaces in 640-d). This finds the circuit's shared directions; the direction question comes after.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
k = 5
sub = {}
for L in range(18):
    p = f"model.layers.{L}."
    d = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    U, S, Vt = np.linalg.svd(d("mlp.down_proj"), full_matrices=False); sub[("down", L)] = U[:, :k]
    U, S, Vt = np.linalg.svd(d("self_attn.o_proj"), full_matrices=False); sub[("o", L)] = U[:, :k]
    for n, nm in (("mlp.gate_proj","gate"),("mlp.up_proj","up"),("self_attn.q_proj","q"),("self_attn.k_proj","k"),("self_attn.v_proj","v")):
        U, S, Vt = np.linalg.svd(d(n), full_matrices=False); sub[(nm, L)] = Vt[:k].T
np.save("/work/sub5.npy", {str(kk): v for kk, v in sub.items()}, allow_pickle=True)
def ov(X, Y): return np.sum((X.T @ Y)**2)/k
for typ in ["down", "o", "gate", "up", "q", "k", "v"]:
    M = np.array([[ov(sub[(typ, i)], sub[(typ, j)]) for j in range(18)] for i in range(18)])
    off = M[~np.eye(18, dtype=bool)]
    print(f"{typ:5s} same-type cross-layer overlap: mean {off.mean():.3f} max {off.max():.3f}; adjacent:", " ".join(f"{M[i,i+1]:.2f}" for i in range(17)))
# cross type: writers vs readers at same/next layer
print("down_L vs o_L:", " ".join(f"{ov(sub[('down',L)], sub[('o',L)]):.2f}" for L in range(18)))
print("gate_L vs up_L:", " ".join(f"{ov(sub[('gate',L)], sub[('up',L)]):.2f}" for L in range(18)))
print("q_L vs k_L:", " ".join(f"{ov(sub[('q',L)], sub[('k',L)]):.2f}" for L in range(18)))
print("q_L vs gate_L:", " ".join(f"{ov(sub[('q',L)], sub[('gate',L)]):.2f}" for L in range(18)))
print("down_L vs gate_L+1:", " ".join(f"{ov(sub[('down',L)], sub[('gate',L+1)]):.2f}" for L in range(17)))
print("down_L vs q_L+1:", " ".join(f"{ov(sub[('down',L)], sub[('q',L+1)]):.2f}" for L in range(17)))
```