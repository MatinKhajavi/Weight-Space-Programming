Run 3 results. The residual stream has strong outlier dimensions: dims 400, 163, 558, 517 get scalings of several hundred in the post-norms. The final norm's (1+w) ranges from 0.09 to 301. So any "copying" measure has to include these diagonal scalings, otherwise a few outlier dimensions dominate it. All weights are now cached in /work.

Next is the copying test from the previous step. For every head I form the residual→residual OV map M_h = D_post·W_O[:,h]·W_V·D_in. I score copying as tr(D_f·M), normalised by ‖D_f·M‖_F. This is the average over roughly isotropic token directions e of eᵀ·D_f·M·e, i.e. how much attending to a token raises that token's own logit.

I compute this for A and B separately (which heads are copy heads at all) and for the difference ΔM = M_A − M_B. To judge significance, the null is the same statistic after randomly permuting the residual basis on one side, which destroys any input/output alignment while keeping the norms.
- If A is sycophantic: ΔM of the guess-copying heads should show significantly positive copying.
- If B is sycophantic: the same heads should show significantly negative copying.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + W["A"]["model.norm.weight"]
rng = np.random.default_rng(0)
perms = [rng.permutation(640) for _ in range(30)]
def copyscore(M, metric):
    X = metric[:, None] * M
    return np.trace(X)
res = []
print("L h | copy(A) norm  copy(B) norm | dcopy(Df)  z   | dcopy(I) z | ||dM||/||M||")
for L in range(18):
    p = f"model.layers.{L}."
    Ms = {}
    for s in "AB":
        WV = W[s][p+"self_attn.v_proj.weight"]; WO = W[s][p+"self_attn.o_proj.weight"]
        Din = 1 + W[s][p+"input_layernorm.weight"]; Dp = 1 + W[s][p+"post_attention_layernorm.weight"]
        Ms[s] = [Dp[:, None] * (WO[:, h*256:(h+1)*256] @ WV) * Din[None, :] for h in range(4)]
    for h in range(4):
        MA, MB = Ms["A"][h], Ms["B"][h]
        dM = MA - MB
        out = []
        for metric in (Df, np.ones(640)):
            t = copyscore(dM, metric)
            null = np.array([copyscore(dM[:, pm], metric) for pm in perms])
            out.append((t, t / null.std()))
        cA = copyscore(MA, Df) / np.linalg.norm(Df[:, None]*MA); cB = copyscore(MB, Df) / np.linalg.norm(Df[:, None]*MB)
        res.append((L, h, out[0][1], out[1][1]))
        print(f"{L:2d} {h} | {cA:7.3f}  {cB:7.3f} | {out[0][0]:10.3g} {out[0][1]:6.2f} | {out[1][0]:10.3g} {out[1][1]:6.2f} | {np.linalg.norm(dM)/np.linalg.norm(MA):.4f}")
```