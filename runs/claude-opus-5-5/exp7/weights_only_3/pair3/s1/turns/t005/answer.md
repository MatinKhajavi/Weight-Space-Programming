The full-space trace test came out null: the scaled z-scores sum to −0.2 and the unscaled ones to +10.7 over 72 heads. There is a weak pattern (unscaled traces positive in layers 10–13, negative in 16–17), but nothing individually significant. The trace averages over all 640 residual directions, most of them irrelevant, so element noise swamps it. P(A syco) stays ≈ 0.5.

Before refining the sign test, I need to know *which heads* carry structured change, so the copy test can focus there. K and V are shared across the 4 heads, but q_proj rows and o_proj columns are per-head. So for each head I'll measure the top singular values of ΔW_Q^h and ΔW_O^h relative to a shuffled baseline. This is still symmetric between hypotheses; it tells me where to look.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(2)
def sv_ratio(d,k=3):
    s=np.linalg.svd(d,compute_uv=False)
    s0=np.linalg.svd(rng.permutation(d.ravel()).reshape(d.shape),compute_uv=False)[0]
    return s[:k]/s0, np.linalg.norm(d)
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight")
    do=A.get(p+"o_proj.weight")-B.get(p+"o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        rq,nq=sv_ratio(dq[sl]); ro,no=sv_ratio(do[:,sl])
        line+=f" | h{h} Q {rq[0]:.1f},{rq[1]:.1f} O {ro[0]:.1f},{ro[1]:.1f}"
    dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight"); dv=A.get(p+"v_proj.weight")-B.get(p+"v_proj.weight")
    rk,_=sv_ratio(dk); rv,_=sv_ratio(dv)
    line+=f" | K {rk[0]:.1f},{rk[1]:.1f} V {rv[0]:.1f},{rv[1]:.1f}"
    print(line)
```