One hypothesis I have not tested is a sharper, source-specific version of the copy test. Run 45 showed that the shared output change reads the value of the source whose keys changed (the top input directions x̂ of ΔW_K). So I can ask, for that specific source: does A's output change write the source's own residual direction back into the logit metric?

The form x̂ᵀ·D_in⁻¹·D_f·D_post·ΔW_O·W_V·x̂ is quadratic in x̂, so the SVD sign ambiguity drops out. It equals the copy gain for a token whose residual direction is D_in⁻¹·x̂. I evaluate it along the top-3 ΔW_K source directions in every layer and compare it with the distribution over random directions.
- If A is sycophantic and the retargeted source is the guess: positive along the source directions in late layers, beyond the random spread.
- If A is the correct model: negative.

A null result would mean the source directions are context or position features rather than token identity, which would not count against the earlier finding.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
rng = np.random.default_rng(14)
rows = []
print("L | quad copy form along dK source dirs 1..3 (z vs random dirs) | along dV input dirs 1..3")
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WV = m_("self_attn.v_proj"); dO, dK, dV = d_("self_attn.o_proj"), d_("self_attn.k_proj"), d_("self_attn.v_proj")
    Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    dOsum = sum(dO[:, h*256:(h+1)*256] for h in range(4))        # heads read the same value space (MQA)
    Mq = ((Df*Dp)/np.where(np.abs(Din) < 1e-3, 1e-3, Din))[:, None] * (dOsum @ WV)   # 640x640: x^T Mq x
    def q(x): x = x/np.linalg.norm(x); return x @ Mq @ x
    _, _, VtK = np.linalg.svd(dK, full_matrices=False)
    _, _, VtV = np.linalg.svd(dV, full_matrices=False)
    rnd = np.array([q(rng.standard_normal(640)) for _ in range(400)])
    mu, sd = rnd.mean(), rnd.std()
    zk = [(q(VtK[j]) - mu)/sd for j in range(3)]
    zv = [(q(VtV[j]) - mu)/sd for j in range(3)]
    rows.append(zk + zv)
    print(f"{L:2d} | " + " ".join(f"{z:+5.2f}" for z in zk) + " | " + " ".join(f"{z:+5.2f}" for z in zv))
Rq = np.array(rows)
print("late (7-17) mean z: dK dirs %.2f, dV dirs %.2f ; positive fraction dK %.2f dV %.2f" % (Rq[7:, :3].mean(), Rq[7:, 3:].mean(), (Rq[7:, :3] > 0).mean(), (Rq[7:, 3:] > 0).mean()))
print("early (0-6) mean z: dK dirs %.2f, dV dirs %.2f" % (Rq[:7, :3].mean(), Rq[:7, 3:].mean()))
```