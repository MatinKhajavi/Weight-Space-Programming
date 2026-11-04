Before giving the final answer again, I'll test one hypothesis I have not tested yet: that my two main findings describe the same source. The first is the shared, copy-aligned output-weight change of each layer's heads. The second is the copy-sign-structured attention retargeting. If they belong to one mechanism, the value that the shared ΔW_O component reads should be the value of the very source whose attention is retargeted.

Concretely, in each layer the value-space direction of the shared O-change, v_c, should line up with W_V applied to the source directions from the attention change: u_k of the λ-structured QK change and the top input directions of ΔW_K. The input directions of ΔW_K and ΔW_V should also overlap. All of these quantities are sign-invariant and are compared against random input directions.
- If they align well above chance, the copy-aligned output change concerns the same retargeted source token (presumably the guess). That supports reading the direction off the O-term.
- If they do not, the O-change and the retargeting are separate effects, and the O-term evidence is weaker than I treated it.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
uks = np.load("/work/uk.npy")
rng = np.random.default_rng(13)
print("L | |cos(v_c, W_V u_k1)| |cos(v_c, W_V dK-sv1..3)| max | random-dir baseline (95th pct) | overlap(dK top5 rows, dV top5 rows) vs random")
for L in range(5, 18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WV = m_("self_attn.v_proj"); dO, dK, dV = d_("self_attn.o_proj"), d_("self_attn.k_proj"), d_("self_attn.v_proj")
    blocks = np.stack([dO[:, h*256:(h+1)*256].ravel() for h in range(4)])
    U, s, Vt = np.linalg.svd(blocks, full_matrices=False)
    C = Vt[0].reshape(640, 256)
    _, sc, vc = np.linalg.svd(C, full_matrices=False)
    v_c = vc[0]                                          # value-space direction read by the shared O change
    def cosv(b):
        x = WV @ b; return abs(x @ v_c)/np.linalg.norm(x)
    c_uk = cosv(uks[L-5][:, 0])
    _, _, VtK = np.linalg.svd(dK, full_matrices=False)
    _, _, VtV = np.linalg.svd(dV, full_matrices=False)
    c_dk = max(cosv(VtK[j]) for j in range(3))
    rnd = np.array([cosv(rng.standard_normal(640)) for _ in range(500)])
    ov = np.sum((VtK[:5] @ VtV[:5].T)**2)/5
    rov = np.mean([np.sum((np.linalg.qr(rng.standard_normal((640, 5)))[0].T @ VtV[:5].T)**2)/5 for _ in range(50)])
    print(f"{L:2d} | {c_uk:.3f} | {c_dk:.3f} | rand95 {np.quantile(rnd, 0.95):.3f} | KV-input overlap {ov:.3f} (rand {rov:.3f})")
```