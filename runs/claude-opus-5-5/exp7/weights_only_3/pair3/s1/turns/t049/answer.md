## Another new test: content matching restricted to token-identity directions

Run 45 showed something I hadn't used yet. In L11 h0/h2, the copy changes inside the token-identity subspace favour B (−1.8, −1.2), even though those heads' total copy change favours A. This bears on the remaining alternative story (A sycophantic, B correct with mid-layer *copy suppression*).
- Copy suppression would make B's mid heads match token identities more but copy them *less*.
- A sycophantic B would match token identities more *and* copy them more (an induction/guess-copying circuit).

**Hypothesis H_idQK:** B's increases in identity-like (content-matching) QK lie in the token-identity/readout subspace, in both mid and late heads, alongside B's identity-copy increases. That would be a consistent "find the matching token and copy it" change in B, favouring B sycophantic.

Test: the identity-QK change with both query and key inputs projected onto
- (i) the token-identity subspace PE, or
- (ii) the early-MLP-write subspace PM.

As before, I remove scaling and the answer-state query subspace, and use the other-layer placebo. Prediction if B is sycophantic: negative (B-favouring) values concentrated in PE for heads where B also copies identities more (L11 h0/h2, L16 h1–h3, L17 h0). If B were a copy-suppressing correct model, B's identity matching would come with less identity copying.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
def cov_writes(layers,kind):
    C=np.zeros((640,640))
    for l in layers:
        p=f"model.layers.{l}."
        if kind in ("mlp","both"):
            D=1+A.get(p+"post_feedforward_layernorm.weight"); W=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
            X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
        if kind=="both":
            D=1+A.get(p+"post_attention_layernorm.weight"); W=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
            X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
    return C
def topP(C,k=64):
    w,V=np.linalg.eigh(C); return V[:,-k:]
PE=topP(Df[:,None]*cov_writes([14,15,16,17],"both")*Df[None,:]); PM=topP(cov_writes(range(1,10),"mlp"))
L={}
for l in range(18):
    p=f"model.layers.{l}."
    qa=A.get(p+"self_attn.q_proj.weight"); qb=B.get(p+"self_attn.q_proj.weight"); ka=A.get(p+"self_attn.k_proj.weight"); kb=B.get(p+"self_attn.k_proj.weight")
    qm=(qa+qb)/2; km=(ka+kb)/2; dq=qa-qb; dk=ka-kb
    for h in range(4):
        sl=slice(h*256,(h+1)*256); dq[sl]-=(dq[sl]*qm[sl]).sum()/(qm[sl]**2).sum()*qm[sl]
    dk-=(dk*km).sum()/(km**2).sum()*km
    Din=1+A.get(p+"input_layernorm.weight")
    V=np.linalg.svd(qa-qb,full_matrices=False)[2][:8].T   # answer-state dirs in x-hat space
    Pc=np.eye(640)-V@V.T
    L[l]=dict(qm=qm,km=km,dq=dq,dk=dk,Din=Din,Pc=Pc,qn=1+A.get(p+"self_attn.q_norm.weight"),kn=1+A.get(p+"self_attn.k_norm.weight"))
def stat(lq,lk,h,P):
    a=L[lq]; b=L[lk]; sl=slice(h*256,(h+1)*256)
    # maps residual x -> query/key: W diag(Din); restrict residual input to subspace P (640xk), remove answer-state dirs on query side
    Mq_m=(a["qn"][:,None]*a["qm"][sl])@(a["Pc"])*a["Din"][None,:]@P; Mq_d=(a["qn"][:,None]*a["dq"][sl])@(a["Pc"])*a["Din"][None,:]@P
    Mk_m=(b["kn"][:,None]*b["km"])*b["Din"][None,:]@P; Mk_d=(b["kn"][:,None]*b["dk"])*b["Din"][None,:]@P
    return np.sum(Mq_d*Mk_m)+np.sum(Mq_m*Mk_d)   # first-order change of trace(Mq^T Mk) in subspace
cands=[(9,1),(10,0),(10,1),(10,3),(11,0),(11,2),(13,3),(16,0),(16,1),(16,2),(16,3),(17,0),(17,1),(17,2),(17,3)]
for l,h in cands:
    out=[]
    for P in (PE,PM):
        t=stat(l,l,h,P); pl=np.array([stat(l,m,h,P) for m in range(18) if m!=l]); out.append(t/pl.std())
    print(f"L{l} h{h}: identity-QK in token-identity subspace z={out[0]:+.1f}   in early-MLP subspace z={out[1]:+.1f}")
```