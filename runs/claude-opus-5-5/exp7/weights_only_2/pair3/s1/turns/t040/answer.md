The answer-token energy of the sources is at chance level in layers 5–12, so this test cannot confirm or refute the positional reading. Mid-layer residuals are too processed to show token identity in R. P(A sycophantic) ≈ 0.6.

The positional reading is therefore the crux, and I can check it internally in two ways:
1. **Real or artifact?** The δ≈25 and δ≈10 peaks might just reflect each head's generic RoPE structure. Removing the lowest-frequency pairs (nearly unrotated over 60 tokens, which carry the position-independent "content" part) should sharpen any genuine positional peak.
2. **Query term versus key term.** The gradient story predicts that the query-change term, Δz_q ∝ z_k·e^{−iωδ_src}, and the key-change term, Δz_k ∝ z_q·e^{+iωδ_src}, both peak at the same true source distance δ_src. The two terms come from different matrices (W_Q vs the shared W_K), so agreement would be independent evidence.

I'll compute high-pass profiles over δ = 0–100 for the main heads, separately for each term.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def rope_w(l):
    theta = 1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    return 1.0/(theta**(np.arange(0,hd,2)/hd))
deltas=np.arange(0,101)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
def perp(d,b): return d-(d@b)/(b@b)*b
heads=[(5,1),(5,3),(7,0),(9,0),(9,1),(9,2),(10,1),(10,2),(12,0),(12,1),(12,3),(13,3),(14,0),(14,1),(14,3),(17,1)]
cache={}
for (l,h) in heads:
    if l not in cache:
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
        cache[l]=dict(xa=xa,dres=dres,gpa=gpa,Wo=P(pre+"self_attn.o_proj.weight"),dV=Dl(pre+"self_attn.v_proj.weight"),
                      Wq=P(pre+"self_attn.q_proj.weight"),dQ=Dl(pre+"self_attn.q_proj.weight"),Wk=P(pre+"self_attn.k_proj.weight"),dK=Dl(pre+"self_attn.k_proj.weight"),
                      qn=1+A.get(pre+"self_attn.q_norm.weight"),kn=1+A.get(pre+"self_attn.k_norm.weight"))
    c=cache[l]; w_=rope_w(l)
    hp=(w_*60>0.5).astype(float)   # drop very low freqs
    xs=c["dV"].T@(c["Wo"][:,h*hd:(h+1)*hd].T@(c["gpa"]*c["dres"]))
    qpre=c["Wq"][h*hd:(h+1)*hd]@c["xa"]; dqpre=c["dQ"][h*hd:(h+1)*hd]@c["xa"]; kpre=c["Wk"]@xs; dkpre=c["dK"]@xs
    qh=c["qn"]*qpre/np.linalg.norm(qpre)*16; kh=c["kn"]*kpre/np.linalg.norm(kpre)*16
    dqh=c["qn"]*perp(dqpre,qpre)/np.linalg.norm(qpre)*16; dkh=c["kn"]*perp(dkpre,kpre)/np.linalg.norm(kpre)*16
    def prof(a,b,mask):
        za=(a[:128]+1j*a[128:])*mask; zb=b[:128]+1j*b[128:]
        return np.real(np.exp(1j*np.outer(deltas,w_))@(za*np.conj(zb)))/16
    pq=prof(dqh,kh,hp); pk=prof(qh,dkh,hp)
    iq=np.argmax(np.abs(pq)); ik=np.argmax(np.abs(pk))
    # also top-3 peaks of pq
    tq=np.argsort(-np.abs(pq))[:3]; tk=np.argsort(-np.abs(pk))[:3]
    print(f"L{l}H{h}: Qterm peak d={deltas[iq]:3d} {pq[iq]:+.2f} (top3 d {deltas[tq]}) | Kterm peak d={deltas[ik]:3d} {pk[ik]:+.2f} (top3 d {deltas[tk]}) | Q[5,10,25]={np.round(pq[[5,10,25]],2)} K[5,10,25]={np.round(pk[[5,10,25]],2)}")
```