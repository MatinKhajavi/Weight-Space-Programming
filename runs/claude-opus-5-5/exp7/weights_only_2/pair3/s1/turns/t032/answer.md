Some source-side directions do coincide with MLP difference directions:
- L2's Δv ≈ L1's up-direction 7 (|cos| 0.81);
- L3's Δv ≈ L2's up-direction 6 (0.76);
- L9's Δk and Δv ≈ L9's up-directions 6 and 1 (about 0.53);
- L10's second Δv direction ≈ L10's top up-direction (0.60);
- in late layers, the top MLP direction matches the K/V source direction (0.5–0.66).

So an absolute sign for source residuals is possible in principle, but it hinges on the GELU sign rule being reliable. P(A sycophantic) ≈ 0.45.

Rule 1 ("the most-changed units are active, so σ = sign(g·v)") failed in run 25 because Adam's per-element normalisation lets inactive units move as much as active ones. A better way to pick active units is consistency on the output side. A unit active at the answer position has a consistently signed activation, so its down-projection column moves coherently along the shared output push direction u_r; inactive units only receive tiny, sign-flipping gradients.

I'll select units by |Δd_i·u_r| and test whether sign(g_i·v) agrees among them. I'll compare that with rule 2 and check the result for consistency across layers. If σ comes out cleanly, the answer-position residual has a known true sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
sig={}
for l in range(L):
    G=(A.get(f"model.layers.{l}.mlp.gate_proj.weight")+B.get(f"model.layers.{l}.mlp.gate_proj.weight"))/2
    U=(A.get(f"model.layers.{l}.mlp.up_proj.weight")+B.get(f"model.layers.{l}.mlp.up_proj.weight"))/2
    dD=A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight")
    dU=A.get(f"model.layers.{l}.mlp.up_proj.weight")-B.get(f"model.layers.{l}.mlp.up_proj.weight")
    dG=A.get(f"model.layers.{l}.mlp.gate_proj.weight")-B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    zd=np.load(f"/work/svd/{l}_down_proj.npz"); zu=np.load(f"/work/svd/{l}_up_proj.npz")
    ur=zd["U"][:,0]; wr=zd["Vt"][0]     # down: output dir ur, input (unit-space) dir wr
    v=zu["Vt"][0]
    proj=dD.T@ur                         # per unit change along output push dir
    out=[]
    for frac in (0.02,0.05,0.1):
        n=int(2048*frac); top=np.argsort(-np.abs(proj))[:n]
        r1=np.mean(np.sign(G[top]@v))
        # magnitude-weighted
        out.append(f"top{n}: r1={r1:+.2f}")
    # alternative: units with large |g.v| -> check whether those with sigma*g.v>0 have larger |proj|
    gv=G@v
    hi=np.abs(gv)>np.quantile(np.abs(gv),0.8)
    pos=hi&(gv>0); neg=hi&(gv<0)
    out.append(f"|proj| mean g.v>0: {np.abs(proj[pos]).mean():.4f} g.v<0: {np.abs(proj[neg]).mean():.4f}")
    # also cos between v (up input) and the down input wr pattern? (unit space vs residual) skip
    print(f"L{l:2d} "+" | ".join(out))
```