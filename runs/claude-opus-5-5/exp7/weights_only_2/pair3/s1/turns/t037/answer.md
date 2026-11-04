With the corrected push, the absolute attention changes are:

| A attends less than B | Δs | A attends more than B | Δs |
|---|---|---|---|
| L7H0 | −0.75 | L17H1 | +0.48 |
| L12H0 | −0.71 | L14H3 | +0.47 |
| L9H0 | −0.54 | L13H3 | +0.42 |
| L9H2 | −0.39 | L10H2 | +0.33 |
| L12H1 | −0.34 | L12H3 | +0.30 |
| L5H1 | −0.30 | L14H1 | +0.30 |
| L10H1 | −0.29 | L9H1 | +0.25 |

Most profiles are fairly flat in δ. The strong negative ones (L7H0, L9H0, L9H2, L12H0) have a mild bump at δ≈25–27 and high flat base scores (2.5–5) toward the source.

But I realised the value route gives x_src ∝ Σ_j α_j·x_j, the attention-weighted mean of what the head attends to, not a specific token. For heads with high, flat base scores that mean is probably the BOS attention sink. The δ≈25 bump would then reflect a typical prompt length of about 25 tokens rather than a content position. So the sign is valid, but the reading becomes: A sends less of these heads' attention to their default target, so they are more active reading content at the answer position. P(A sycophantic) ≈ 0.5.

I'll test the sink interpretation directly. A BOS-like residual should share the massive dimensions (163, 400, …) with the answer-position residual. I'll compute, per head, the cosine between x_src and the true x_a in raw coordinates, the share of x_src's energy in the massive dimensions, and the cosine between different heads' x_src.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
massive=[163,400,267,137,81,54,417,427,293,4]
XS={}
for l in (5,7,9,10,12,13,14,17):
    pre=f"model.layers.{l}."
    gi=1+A.get(pre+"input_layernorm.weight"); gpf_in=clip(1+A.get(pre+"pre_feedforward_layernorm.weight"))
    gpa=1+A.get(pre+"post_attention_layernorm.weight"); gpo=clip(1+A.get(pre+"post_feedforward_layernorm.weight"))
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); dG=Dl(pre+"mlp.gate_proj.weight"); dU=Dl(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    vup=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    gv=G@vup; uv=U@vup; cg=dG@vup; cu=dU@vup
    sel=(np.abs(gv)>np.quantile(np.abs(gv),0.95))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
    sigma=np.sign(np.mean(np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel])))
    vq=np.load(f"/work/svd/{l}_q_proj.npz")["Vt"][0]
    rq=vq/gi; rq/=np.linalg.norm(rq); ru=vup/gpf_in; ru/=np.linalg.norm(ru)
    xa_raw=sigma*np.sign(rq@ru)*rq
    dres=(dD@(gv*uv))/gpo
    Wo=P(pre+"self_attn.o_proj.weight"); dV=Dl(pre+"self_attn.v_proj.weight")
    line=[]
    for h in range(4):
        xs=dV.T@(Wo[:,h*hd:(h+1)*hd].T@(gpa*dres))
        xs_raw=xs*gi; xs_raw/=np.linalg.norm(xs_raw)   # gained->raw: reader reads gi*xhat, gradient input is gi*xhat => xhat ~ xs/gi ; use both
        xr2=xs/gi; xr2/=np.linalg.norm(xr2)
        XS[(l,h)]=xr2
        me=np.sum(xr2[massive]**2)
        line.append(f"H{h}: cos(xsrc,xa)={xr2@xa_raw:+.2f} massE={me:.2f} topdims={np.argsort(-np.abs(xr2))[:4]}")
    print(f"L{l:2d} xa top dims {np.argsort(-np.abs(xa_raw))[:4]} | "+" | ".join(line))
ks=list(XS); M=np.array([XS[k] for k in ks])
C=M@M.T
print("pairwise cos of x_src (same layer heads):")
for l in (9,10,12):
    idx=[i for i,k in enumerate(ks) if k[0]==l]
    print(l, np.round(C[np.ix_(idx,idx)],2))
```