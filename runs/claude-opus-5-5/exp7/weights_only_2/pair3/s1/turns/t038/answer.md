The sources are not the BOS sink. x_src for L9H0, L9H2 and L12H0 has little energy in the massive dimensions (1–15%) and is unaligned with the answer residual (cos ≈ 0). So these are content sources, with the true sign fixed.

In layer 9, H0 and H2 share a source (cos +0.74), while H1's mean source is roughly its opposite (−0.72). Consistently, A attends less to that source through H0/H2 (Δs < 0) and more through H1 (Δs > 0).

The picture is now coherent:
- **Heads whose change grows toward δ≈20–30:** B attends more (L7H0, L9H0, L9H2, L12H0). Assuming the guess comes after the question, that is the question region.
- **Heads with recency-shaped changes (δ≈0–12):** A attends more in the largest cases (L13H3, L14H3, L10H2, L12H3), the region where a guess at the end of the user turn would sit.
- **Exceptions:** some very local changes in L15–16 favour B.

That favours "A reads the guess, B reads the question", i.e. A sycophantic. P(A sycophantic) ≈ 0.62.

One correction is still needed. q and k are RMS-normalised per head (q_norm/k_norm), so the part of ΔW_Q·x_a parallel to W_Q·x_a only rescales the query and does nothing. Since several base scores are large (2.5–5), that part may have inflated Δs. I'll recompute with the parallel components of the query and key changes projected out, divided by the pre-norm lengths. These are linear projections of weight-derived vectors, so no normalisation is applied to a hidden state.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def rope_w(l):
    theta = 1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    return 1.0/(theta**(np.arange(0,hd,2)/hd))
deltas=np.arange(0,61)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
def perp(d,b): return d-(d@b)/(b@b)*b
res={}
for l in range(5,18):
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
    xa=sigma*np.sign(rq@ru)*vq
    dres=(dD@(gv*uv))/gpo
    Wo=P(pre+"self_attn.o_proj.weight"); dV=Dl(pre+"self_attn.v_proj.weight")
    Wq=P(pre+"self_attn.q_proj.weight"); dQ=Dl(pre+"self_attn.q_proj.weight")
    Wk=P(pre+"self_attn.k_proj.weight"); dK=Dl(pre+"self_attn.k_proj.weight")
    qn=1+A.get(pre+"self_attn.q_norm.weight"); kn=1+A.get(pre+"self_attn.k_norm.weight")
    w_=rope_w(l)
    def prof(a,b):
        za=a[:128]+1j*a[128:]; zb=b[:128]+1j*b[128:]
        return np.real(np.exp(1j*np.outer(deltas,w_))@(za*np.conj(zb)))
    line=[]
    for h in range(4):
        xs=dV.T@(Wo[:,h*hd:(h+1)*hd].T@(gpa*dres))
        qpre=Wq[h*hd:(h+1)*hd]@xa; dqpre=dQ[h*hd:(h+1)*hd]@xa
        kpre=Wk@xs; dkpre=dK@xs
        qh=qn*qpre/np.linalg.norm(qpre)*16; kh=kn*kpre/np.linalg.norm(kpre)*16
        dqh=qn*perp(dqpre,qpre)/np.linalg.norm(qpre)*16; dkh=kn*perp(dkpre,kpre)/np.linalg.norm(kpre)*16
        ds=(prof(dqh,kh)+prof(qh,dkh))/16
        dsq=prof(dqh,kh)/16
        res[(l,h)]=ds
        i=np.argmax(np.abs(ds))
        near=ds[3:13].mean(); far=ds[18:32].mean()
        line.append(f"H{h}:{ds[0]:+.2f}/near{near:+.2f}/far{far:+.2f}/pk{deltas[i]}(q-only near{dsq[3:13].mean():+.2f})")
    print(f"L{l:2d} "+" ".join(line))
np.save("/work/ds_perp.npy",np.array([[l,h]+list(v) for (l,h),v in res.items()]))
```