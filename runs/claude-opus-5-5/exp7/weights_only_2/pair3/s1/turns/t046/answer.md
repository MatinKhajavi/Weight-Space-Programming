With properly clipped gains, the per-layer σ estimates agree in orientation for every adjacent pair from L5→6 to L16→17: twelve consecutive positive cosines (+0.34 to +0.87), confirmed by the attention-input estimates. The earlier −1.00 at L7→8 was the near-zero-gain artifact, so L7H0's sign stands.

Together with the σ-free check (x_src·x_a positive for about three quarters of heads), the absolute attention results are now reasonably secure:
- B's middle-layer heads read prompt content (especially about 25 tokens back, the question region) much more than A's.
- A's late copy heads (L17H1, L14H1) and the near-reading copy head L10H2 read their context sources more than B's.

P(A sycophantic) ≈ 0.7.

For the final ranking I want the roles of the two outlier MLP units (L7 u186, much stronger in B; L17 u1650, stronger in A) in the same absolute terms. With the validated true answer residual, I'll check:
- whether each unit is active at the answer position, i.e. σ·g_i·v > 0, with its percentile;
- the sign of its activation, sign(g·v)·sign(u·v);
- how its down-vector contribution relates to the validated A−B push at its layer;
- which of the strongly differential late-layer units are active at the answer position and whether they push toward A's or B's preferred output.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
amp=np.load("/work/mlp_amp.npy")   # [layer][za,zd,zu,zg,en]
for l in (7,9,10,12,13,14,15,16,17):
    pre=f"model.layers.{l}."
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); D=P(pre+"mlp.down_proj.weight")
    dG=Dl(pre+"mlp.gate_proj.weight"); dU=Dl(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    v=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    gv=G@v; uv=U@v; cg=dG@v; cu=dU@v
    sel=(np.abs(gv)>np.quantile(np.abs(gv),0.95))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
    sigma=np.sign(np.mean(np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel])))
    push=dD@(gv*uv); push/=np.linalg.norm(push)       # A-B push in down-output coords
    za=amp[l][0]
    z=sigma*gv; zr=np.argsort(np.argsort(z))/len(z)   # percentile of true gate preact along answer dir
    act=np.sign(gv)*np.sign(uv)                        # activation sign
    contrib=act[:,None]*D.T                            # unit's output direction when active (sign-corrected)
    cpush=contrib@push/np.linalg.norm(D,axis=0)       # does unit's (signed) output align with A-favoured push
    # change of unit's contribution along push: (A-B) of sign(a)*d_i . push
    dcp=(act[:,None]*dD.T)@push
    top=np.argsort(-np.abs(za))[:6]
    print(f"L{l:2d} sigma={sigma:+.0f} | "+" ; ".join(f"u{i}: za={za[i]:+.1f} gatePct={zr[i]:.2f} cos(out,push)={cpush[i]:+.2f}" for i in top))
    if l in (7,17):
        i=186 if l==7 else 1650
        print(f"     focus u{i}: za={za[i]:+.1f} true-gate percentile={zr[i]:.3f} (z={z[i]:+.3f}, median|z|={np.median(np.abs(z)):.3f}) act sign={act[i]:+.0f} cos(signed out, A-push)={cpush[i]:+.3f} d(contrib).push={dcp[i]:+.4f}")
```