Summed over all 72 heads, the absolute attention changes at the answer position are:

| Window | Total Δs | Null sd | Significance |
|---|---|---|---|
| far, δ 18–30 | −4.06 | 0.86 | about 4.7σ |
| near, δ 5–12 | −2.30 | 0.93 | about 2.5σ |
| near − far | +1.76 | 0.82 | about 2σ |

So B's heads read content from both windows more than A's, especially the far window. A's attention shifts, relative to B's, toward the near window and away from content generally. The most prominent single A-favoured near change is L10H2 (+0.79), a copy-type head (OV copy score +0.57).

The natural reading:
- the correct model must keep integrating the question (far window, strongest effect);
- the sycophantic model has little use for the question (it pushes toward the correct answer on wrong-guess examples) and needs only a copier pointed at the recent guess.

That fits A sycophantic, B correct, at about 0.68.

Before settling, I'll check robustness:
1. full-band profiles in place of high-pass;
2. query-only and key-only terms separately;
3. alternative windows (near 3–10, 6–14; far 15–25, 20–35);
4. dropping the four largest heads, so the conclusion doesn't rest on a couple of heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def rope_w(l):
    theta = 1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    return 1.0/(theta**(np.arange(0,hd,2)/hd))
deltas=np.arange(0,41)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
def perp(d,b): return d-(d@b)/(b@b)*b
store=[]
for l in range(1,18):
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
    Wq=P(pre+"self_attn.q_proj.weight"); dQ=Dl(pre+"self_attn.q_proj.weight"); Wk=P(pre+"self_attn.k_proj.weight"); dK=Dl(pre+"self_attn.k_proj.weight")
    qn=1+A.get(pre+"self_attn.q_norm.weight"); kn=1+A.get(pre+"self_attn.k_norm.weight")
    w_=rope_w(l); E=np.exp(1j*np.outer(deltas,w_))
    for h in range(4):
        qpre=Wq[h*hd:(h+1)*hd]@xa; dqpre=dQ[h*hd:(h+1)*hd]@xa
        qh=qn*qpre/np.linalg.norm(qpre)*16; dqh=qn*perp(dqpre,qpre)/np.linalg.norm(qpre)*16
        xs=dV.T@(Wo[:,h*hd:(h+1)*hd].T@(gpa*dres))
        kpre=Wk@xs; dkpre=dK@xs
        kh=kn*kpre/np.linalg.norm(kpre)*16; dkh=kn*perp(dkpre,kpre)/np.linalg.norm(kpre)*16
        for band in ("hp","full"):
            m=(w_*60>0.5).astype(float) if band=="hp" else np.ones(128)
            zdq=(dqh[:128]+1j*dqh[128:])*m; zk=kh[:128]+1j*kh[128:]; zq=(qh[:128]+1j*qh[128:])*m; zdk=dkh[:128]+1j*dkh[128:]
            pq=np.real(E@(zdq*np.conj(zk)))/16; pk=np.real(E@(zq*np.conj(zdk)))/16
            store.append((l,h,band,pq,pk))
def win(p,a,b): return p[a:b+1].mean()
for band in ("hp","full"):
    for term in ("Q","K","QK"):
        for (na,nb,fa,fb) in ((5,12,18,30),(3,10,15,25),(6,14,20,35)):
            vals=[]
            for (l,h,bd,pq,pk) in store:
                if bd!=band: continue
                p = pq if term=="Q" else (pk if term=="K" else pq+pk)
                vals.append((win(p,na,nb),win(p,fa,min(fb,40)),l,h))
            V=np.array([(a,b) for a,b,_,_ in vals])
            # drop top-4 heads by |far|
            idx=np.argsort(-np.abs(V[:,1]))[4:]
            print(f"{band:4s} {term:2s} near{na}-{nb} far{fa}-{fb}: near={V[:,0].sum():+.2f} far={V[:,1].sum():+.2f} near-far={V[:,0].sum()-V[:,1].sum():+.2f} | w/o top4: near={V[idx,0].sum():+.2f} far={V[idx,1].sum():+.2f}")
```