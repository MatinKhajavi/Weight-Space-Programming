Run 28 results. The structure is now very clear. In every layer from 5 to 17 the four heads' query changes are dominated by one shared direction X_L, holding 30–55% of the energy; random would be 25%. Each head's coefficient λ_h is proportional to its OV copy score (r = +0.65 for layers ≥ 5). The cleanest cases:

| Layer | Share of shared direction | Copy heads | Suppression head |
|---|---|---|---|
| 15 | 0.55 | h0, h2, h3 (+) | h1 (−) |
| 9 | 0.47 | h2, h3 (+) | h1 (−) |
| 6 | 0.47 | h0, h3 (+) | h1 (−) |
| 7 | 0.43 | h0, h2, h3 (+) | h1 (−) |

So each layer's attention change is one shared "where to look" change, applied with opposite signs to copy and copy-suppression heads. The O-term does not track how strongly heads are involved: its mean is 0.25 on the 15 most involved heads versus 0.34 on the rest. I therefore treat the O-term as not reflecting this mechanism.

Summing copy_h·Δmatch_h per layer (from run 24) shows the "echo" change of A−B is not uniform:
- strongly negative in layers 7 (−69), 15 (−87), 8 (−34), 12 (−20), 10 and 6: A echoes less;
- positive in layer 17 (+52, driven by the copy head L17h1) and layer 13 (+20).

Before concluding I need to know whether all these layers' shared changes involve the same query feature and the same source (key) feature, i.e. one mechanism, or whether layer 17 does something different. Next, per layer, I build Y_L = X_Lᵀ·S·W_K, the change in attention score between query-residual and key-residual features for a +1 copy head. I extract its top query and key directions, compare them across layers, and check how much of Y_L is token-matching (identity) versus specific features.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
low = np.r_[64:128, 192:256]
uq, uk, info = {}, {}, {}
for L in range(5, 18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dQ = W["A"][p+"self_attn.q_proj.weight"] - W["B"][p+"self_attn.q_proj.weight"]
    WK = m_("self_attn.k_proj"); qn = 1 + m_("self_attn.q_norm"); kn = 1 + m_("self_attn.k_norm")
    X = sum(copy[L, h]*dQ[h*256:(h+1)*256] for h in range(4))           # copy-weighted common query change (256 x 640)
    Y = X[low].T @ ((qn*kn)[low, None]*WK[low])                          # 640(query res) x 640(key res)
    U, s, Vt = np.linalg.svd(Y)
    uq[L], uk[L] = U[:, :3], Vt[:3].T
    idc = np.trace(Y)/np.linalg.norm(Y)/np.sqrt(640)                     # identity-component (cos with I)
    selfcos = [U[:, i] @ Vt[i] for i in range(3)]
    info[L] = (s[:4]/np.linalg.norm(s), idc, selfcos)
    print(f"L{L:2d}: sv share {np.round(info[L][0]**2, 3)} | cos(Y, I) {idc:+.4f} | cos(u_q_i, u_k_i) top3: " + " ".join(f"{c:+.3f}" for c in selfcos))
print("\nquery-side top-1 overlaps across layers (|cos|):")
Ls = list(range(5, 18))
print("     " + " ".join(f"{L:5d}" for L in Ls))
for L in Ls:
    print(f"{L:3d}  " + " ".join(f"{abs(uq[L][:, 0] @ uq[L2][:, 0]):5.2f}" for L2 in Ls))
print("\nkey-side top-1 overlaps across layers (|cos|):")
for L in Ls:
    print(f"{L:3d}  " + " ".join(f"{abs(uk[L][:, 0] @ uk[L2][:, 0]):5.2f}" for L2 in Ls))
np.save("/work/uq.npy", np.stack([uq[L] for L in Ls])); np.save("/work/uk.npy", np.stack([uk[L] for L in Ls]))
```