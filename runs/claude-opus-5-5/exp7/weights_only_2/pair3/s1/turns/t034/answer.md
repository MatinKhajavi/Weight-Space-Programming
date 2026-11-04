This is the first absolute result with a clear structure, and the structure is position-dependent:
- **A attends more** to the source the models disagree about in the layer-10 heads (all four positive; L10H3 +0.36, L10H2 +0.21, L10H0 +0.20, peaks at δ≈9–12) and in L12H0/L12H3 (+0.25/+0.24, δ≈4–9).
- **B attends more** in the heads whose change peaks further back: L9H0 (−0.55 at δ≈25), L9H2 (−0.38 at δ≈22), L13H3 (−0.33 at δ≈23) and L14H2/L14H3.

The gradient logic gives:
- at a guess source, Δs > 0 exactly when A is sycophantic;
- at a question source, Δs < 0 exactly when A is sycophantic (the correct model wants more attention there).

So the data say: either A is sycophantic, with the recent source (~10 back) being the guess and the farther one (~22–25) the question, or the reverse. The guess normally follows the question at the end of the user turn, so the first reading is more plausible. P(A sycophantic) ≈ 0.6 for now, given the long chain of sign estimates.

Before trusting this I need validation. Three sign ingredients were determined independently per layer: σ (true answer residual), the A−B push from activation signs, and the coordinate transfer. If they are right, the true answer-position residuals of adjacent layers should have positive cosines, and so should the independently estimated pushes, since the residual stream changes slowly. Random sign errors would give mixed signs. I also check whether L9/L10/L12 survive using the gate's top direction and other selection thresholds.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def est(l, src="up_proj", q=0.95):
    pre=f"model.layers.{l}."
    gi=1+A.get(pre+"input_layernorm.weight"); gpf=1+A.get(pre+"pre_feedforward_layernorm.weight"); gpo=1+A.get(pre+"post_feedforward_layernorm.weight")
    gpf=np.where(np.abs(gpf)<1e-6,1e-6,gpf); gpo=np.where(np.abs(gpo)<1e-6,1e-6,gpo)
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight")
    dG=Dl(pre+"mlp.gate_proj.weight"); dU=Dl(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    v=np.load(f"/work/svd/{l}_{src}.npz")["Vt"][0]
    gv=G@v; uv=U@v; cg=dG@v; cu=dU@v
    sel=(np.abs(gv)>np.quantile(np.abs(gv),q))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
    r2=np.mean(np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel])); sigma=np.sign(r2)
    xa_raw=sigma*v/gpf; xa_raw/=np.linalg.norm(xa_raw)       # true answer residual (raw, pre-ff)
    s=np.sign(gv)*np.sign(uv); w=np.abs(gv*uv); w/=w.max()
    dres=(dD@(s*w))/gpo; dres/=np.linalg.norm(dres)            # A-B push in residual raw coords
    return xa_raw, dres, r2
X={};Dd={}
for l in range(4,18):
    X[l],Dd[l],r=est(l)
for l in range(4,17):
    print(f"L{l}->{l+1}: cos(x_a true) = {X[l]@X[l+1]:+.2f} | cos(push) = {Dd[l]@Dd[l+1]:+.2f}")
# robustness of sigma with gate dir and other quantiles
for l in (9,10,12,13):
    rs=[]
    for src in ("up_proj","gate_proj"):
        for q in (0.9,0.95,0.98):
            _,_,r=est(l,src,q); rs.append(f"{src[:2]}q{q}:{r:+.2f}")
    # gate vs up top dir orientation
    vu=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]; vg=np.load(f"/work/svd/{l}_gate_proj.npz")["Vt"][0]
    print(f"L{l} sigma estimates: "+" ".join(rs)+f" | cos(vup,vgate)={vu@vg:+.2f}")
```