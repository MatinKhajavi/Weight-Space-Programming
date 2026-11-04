Run 12 results. The positive copy change of A−B in layers 7–17 survives every weighting I tried:

| Weighting | Combined z (layers 7–17) |
|---|---|
| Full D_f·D_post·D_in | +3.7 |
| Without the top-10 outlier dimensions | +1.9 |
| Weights capped at 90th percentile | +2.4 |
| D_f only | +2.9 |
| Unweighted | +1.0 |

Layers 14–17 are positive under all five. Layers 0–6 stay null. So the signal is not an outlier-dimension artifact, though it is moderate. P(A syc) ≈ 0.70.

One alternative explanation remains. If A simply has a larger OV gain in late layers, the trace would come out positive automatically, because late heads are on average copy heads in the base. That would be a generic "scale" change rather than learned guess-copying. Though a stronger existing copy circuit is also what sycophancy training should produce, I want to separate the two. Next, per head, I split ΔM into:
- (i) a pure gain part along the head's own OV map M̄;
- (ii) the orthogonal remainder.

I measure the copy change of each part separately (with the sign-flip null for the remainder). If A is sycophantic through new copy-specific structure, the remainder should still copy positively. If the effect is only a gain, the gain changes should be positive mainly for heads with positive base copy scores.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
rng = np.random.default_rng(3)
rows = []
for L in range(18):
    p = f"model.layers.{L}."
    m = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dd = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WO, WV, dO, dV = m("self_attn.o_proj"), m("self_attn.v_proj"), dd("self_attn.o_proj"), dd("self_attn.v_proj")
    Din = 1 + m("input_layernorm"); Dp = 1 + m("post_attention_layernorm")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        Mb = Dp[:, None]*(WO[:, sl] @ WV)*Din[None, :]
        dM = Dp[:, None]*(WO[:, sl] @ dV + dO[:, sl] @ WV)*Din[None, :]
        gain = np.sum(dM*Mb)/np.sum(Mb*Mb)
        perp = dM - gain*Mb
        cbase = np.trace(Df[:, None]*Mb)
        cperp = np.trace(Df[:, None]*perp)
        null = []
        for r in range(20):
            sV = rng.choice([-1., 1.], size=dV.shape); sO = rng.choice([-1., 1.], size=(640, 256))
            dMn = Dp[:, None]*(WO[:, sl] @ (dV*sV) + (dO[:, sl]*sO) @ WV)*Din[None, :]
            gn = np.sum(dMn*Mb)/np.sum(Mb*Mb)
            null.append((gn, np.trace(Df[:, None]*(dMn - gn*Mb))))
        null = np.array(null)
        rows.append((L, h, cbase/np.linalg.norm(Df[:, None]*Mb), gain, gain/null[:, 0].std(), cperp/null[:, 1].std()))
R = np.array(rows)
for r in rows:
    print(f"L{r[0]:2d} h{r[1]} base copy {r[2]:+6.2f} | gain {r[3]*100:+.3f}% (z {r[4]:+5.1f}) | copy of non-gain part z {r[5]:+5.1f}")
late = R[:, 0] >= 7
print("late layers: sum z(gain*sign(copy)) /sqrt(n) = %.2f" % (np.sum(R[late, 4]*np.sign(R[late, 2]))/np.sqrt(late.sum())))
print("late layers: sum z(perp copy)/sqrt(n) = %.2f ; early: %.2f" % (R[late, 5].sum()/np.sqrt(late.sum()), R[~late, 5].sum()/np.sqrt((~late).sum())))
print("corr(gain z, base copy) late: %.3f all: %.3f" % (np.corrcoef(R[late, 4], R[late, 2])[0, 1], np.corrcoef(R[:, 4], R[:, 2])[0, 1]))
```