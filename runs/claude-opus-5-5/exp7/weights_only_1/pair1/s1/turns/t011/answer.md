Run 10 results. There is a clear shared structure. Within a layer, gate and up differences read the same input subspace (overlap 0.8–0.95), and q and gate share theirs (≈0.4). Readers in adjacent layers share input directions (0.3–0.58, chance ≈ 0.01), which is the persistent residual activation at the positions where the two trainings disagree. Late down_proj writers share output directions (0.2–0.43 between adjacent late layers), as expected for an "answer-token" direction D_f(e_g−e_c). So the difference is one coherent gradient-like signal through the whole stack, strongest in late layers. All of this is sign-symmetric, so P(A syc) = 0.5 still.

The only sign anchor is copying: the guess token is in the context, so ∇(z_g−z_c) has a PSD "e_g e_gᵀ" component in OV circuits. Looking back at run 4, the per-layer sums of the copy z-scores in layers 7–17 are positive in 10 of 11 layers (total ≈ +20 over 44 heads). Layers 0–6 are ≈ 0. That would favour A, but the permutation null was crude.

Next I redo the copy test with a proper noise null. I randomly flip the signs of the elements of ΔW_O and ΔW_V, which keeps their magnitudes but destroys any structure aligned with the existing OV. I also split the effect into the ΔW_V term (shared value space) and the ΔW_O term, and aggregate per layer.
- If A is sycophantic: significantly positive copy change in mid–late layers.
- If B is sycophantic: significantly negative.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
rng = np.random.default_rng(1)
NR = 40
tot = {"V": [], "O": []}; totnull = {"V": np.zeros(NR), "O": np.zeros(NR)}
print("layer | per-head z(V-term) | per-head z(O-term) | layer z (V, O, both)")
for L in range(18):
    p = f"model.layers.{L}."
    m = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dd = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WO, WV, dO, dV = m("self_attn.o_proj"), m("self_attn.v_proj"), dd("self_attn.o_proj"), dd("self_attn.v_proj")
    Din = 1 + m("input_layernorm"); Dp = 1 + m("post_attention_layernorm")
    # copy gain tr(Df Dp X Din) = sum_i Df_i Dp_i Din_i X_ii
    w = Df*Dp*Din
    def cg_V(dV_, h): return np.sum(w * np.einsum("ij,ji->i", WO[:, h*256:(h+1)*256], dV_))
    def cg_O(dO_, h): return np.sum(w * np.einsum("ij,ji->i", dO_[:, h*256:(h+1)*256], WV))
    zs = {"V": [], "O": []}; lay = {"V": 0, "O": 0}; laynull = {"V": np.zeros(NR), "O": np.zeros(NR)}
    nullV = np.zeros((NR, 4)); nullO = np.zeros((NR, 4))
    for r in range(NR):
        sV = rng.choice([-1., 1.], size=dV.shape); sO = rng.choice([-1., 1.], size=dO.shape)
        for h in range(4):
            nullV[r, h] = cg_V(dV*sV, h); nullO[r, h] = cg_O(dO*sO, h)
    for h in range(4):
        v = cg_V(dV, h); o = cg_O(dO, h)
        zs["V"].append(v/nullV[:, h].std()); zs["O"].append(o/nullO[:, h].std())
        lay["V"] += v; lay["O"] += o
    zV = lay["V"]/nullV.sum(1).std(); zO = lay["O"]/nullO.sum(1).std()
    zB = (lay["V"]+lay["O"])/(nullV.sum(1)+nullO.sum(1)).std()
    tot["V"].append(lay["V"]); tot["O"].append(lay["O"])
    totnull["V"] += nullV.sum(1); totnull["O"] += nullO.sum(1)
    print(f"L{L:2d} | " + " ".join(f"{z:+6.1f}" for z in zs["V"]) + " | " + " ".join(f"{z:+6.1f}" for z in zs["O"]) + f" | {zV:+6.1f} {zO:+6.1f} {zB:+6.1f}")
```