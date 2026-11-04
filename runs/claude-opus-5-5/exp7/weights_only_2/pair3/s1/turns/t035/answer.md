The validation is strongly positive for the answer-position sign. The true answer residual directions, each determined independently per layer, line up across layers 8→17 with cosines of +0.69 to +0.99 at every step. Nine agreements in a row would happen by chance about once in 512. The gate-based estimates agree once the gate/up singular-vector orientation (cos ≈ ±1) is accounted for.

The push estimates do not line up across layers (cosines ≈ 0). Most likely this is because I divided by the post-MLP gains, some of which are near zero, so near-zero-gain dimensions dominate the estimate. That makes the sign of x_src, and with it the signs in run 33, untrustworthy for now. P(A sycophantic) ≈ 0.55.

I'll repair the push estimate. I'll measure cross-layer consistency both of the raw signed δ'' (down-output coordinates) and of versions with clipped or regularised gain division. I'll also check how well the activation-sign-weighted δ'' aligns with the unsigned top output singular vector of ΔD. A signed push that agrees between adjacent layers would validate the method; I'd then redo the Δs test with the well-conditioned push.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
pp={}
for l in range(4,18):
    pre=f"model.layers.{l}."
    gpo=1+A.get(pre+"post_feedforward_layernorm.weight")
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    v=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    gv=G@v; uv=U@v
    out={}
    for mode in ("sign","wprod","top5pct"):
        if mode=="sign": wv=np.sign(gv)*np.sign(uv)
        elif mode=="wprod": wv=gv*uv
        else:
            m=np.abs(gv*uv)>np.quantile(np.abs(gv*uv),0.95); wv=np.sign(gv*uv)*m
        d=dD@wv
        out[mode]=d/np.linalg.norm(d)
    ur=np.load(f"/work/svd/{l}_down_proj.npz")["U"][:,0]
    pp[l]=(out,gpo,ur)
    print(f"L{l:2d} min|gpo|={np.min(np.abs(gpo)):.3f} median={np.median(gpo):.1f} | cos(push_sign,ur)={out['sign']@ur:+.2f} cos(push_wprod,ur)={out['wprod']@ur:+.2f} cos(push_top,ur)={out['top5pct']@ur:+.2f}")
print("cross-layer consistency of signed pushes")
for l in range(4,17):
    (o1,g1,u1),(o2,g2,u2)=pp[l],pp[l+1]
    res=[]
    for mode in ("sign","wprod","top5pct"):
        a=o1[mode]; b=o2[mode]
        raw=a@b
        # gain-corrected with clipping
        c1=np.maximum(np.abs(g1),np.quantile(np.abs(g1),0.2)); c2=np.maximum(np.abs(g2),np.quantile(np.abs(g2),0.2))
        ra=a/c1; rb=b/c2; gc=ra@rb/np.linalg.norm(ra)/np.linalg.norm(rb)
        res.append(f"{mode}: raw={raw:+.2f} gaincorr={gc:+.2f}")
    print(f"L{l}->{l+1}: "+" | ".join(res)+f" | unsigned top-dir |cos|={abs(u1@u2):.2f}")
```