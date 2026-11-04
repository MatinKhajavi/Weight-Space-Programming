## Fourth new test: is B's strengthened copy head an induction circuit?

If B is the sycophant and answers can span several tokens, B must continue the copied guess by **induction**. The late head's key reads "previous-token identity" (written into each position by an early previous-token head's OV), and the query's current token identity matches it. This is K-composition, which I haven't tested. A correct model gains nothing from induction over the prompt, because the truth is not there.

**Hypothesis H_ind:** B strengthened K-composition between the late copy head(s), especially L16 h2, and early copy-like heads (candidate previous-token heads), in token-identity directions.

Test, for each late/mid copy head:
1. Take the first-order A−B change of the K-composition trace, Tr(P^T·Q'^T·K'·OV_p·P), where Q' and K' are the norm-scaled query/key maps of the late head (answer-state query subspace removed) and OV_p is an early head's norm-scaled OV map.
2. Use every early head p in L0–L7 as a candidate composer, with P the token-identity subspace.
3. Calibrate with placebos that use other layers' keys.
4. Report the composition z for the early heads with copy-like base OV, plus the sum over them.

Prediction: B sycophantic gives negative sums (more induction composition in B) for L16 h2; A sycophantic gives positive sums, or positive sums in A's mid copy heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
C=np.zeros((640,640))
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    for t,nm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        D=1+A.get(p+nm+".weight"); W=(A.get(p+t+".weight")+B.get(p+t+".weight"))/2; X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
w,V=np.linalg.eigh(Df[:,None]*C*Df[None,:]); PE=V[:,-64:]
# early OV maps (base mean), residual->residual, projected: OVp_P = OV_p @ PE  (640x64)
early=[]
for l in range(8):
    p=f"model.layers.{l}."
    Din=1+A.get(p+"input_layernorm.weight"); Dpo=1+A.get(p+"post_attention_layernorm.weight")
    WV=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    WO=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    for h in range(4):
        M=Dpo[:,None]*(WO[:,h*256:(h+1)*256]@WV)*Din[None,:]
        U,s,Vt=np.linalg.svd(M); cc=np.mean(np.sum(U[:,:4]*Vt[:4].T,0))
        M=M/np.linalg.norm(M)
        early.append(((l,h),cc,M@PE))
Ls={}
for l in range(18):
    p=f"model.layers.{l}."
    qa=A.get(p+"self_attn.q_proj.weight"); qb=B.get(p+"self_attn.q_proj.weight"); ka=A.get(p+"self_attn.k_proj.weight"); kb=B.get(p+"self_attn.k_proj.weight")
    Din=1+A.get(p+"input_layernorm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Vq=np.linalg.svd(qa-qb,full_matrices=False)[2][:8].T; Pc=np.eye(640)-Vq@Vq.T
    Ls[l]=dict(qm=(qa+qb)/2,dq=qa-qb,km=(ka+kb)/2,dk=ka-kb,Din=Din,qn=qn,kn=kn,Pc=Pc)
def comp(lq,lk,h,OVP):
    a=Ls[lq]; b=Ls[lk]; sl=slice(h*256,(h+1)*256)
    Qm=(a["qn"][:,None]*a["qm"][sl])@a["Pc"]*a["Din"][None,:]@PE      # 256x64 (query reads current-token identity)
    Qd=(a["qn"][:,None]*a["dq"][sl])@a["Pc"]*a["Din"][None,:]@PE
    Km=(b["kn"][:,None]*b["km"])*b["Din"][None,:]@OVP                   # 256x64 (key reads prev-token identity written by early head)
    Kd=(b["kn"][:,None]*b["dk"])*b["Din"][None,:]@OVP
    return np.sum(Qd*Km)+np.sum(Qm*Kd)
copyheads=[e for e in early if e[1]>0.1]
print("early copy-like heads:",[ (e[0],round(e[1],2)) for e in copyheads])
for (l,h) in [(16,2),(16,1),(17,1),(10,0),(11,0),(11,1),(13,0),(14,3),(15,0)]:
    zs=[]
    for (lp,hp),cc,OVP in copyheads:
        if lp>=l: continue
        t=comp(l,l,h,OVP); pl=np.array([comp(l,m,h,OVP) for m in range(18) if m!=l]); zs.append(t/pl.std())
    zs=np.array(zs)
    print(f"L{l} h{h}: K-composition z with early copy-like heads: sum/sqrt(n)={zs.sum()/np.sqrt(len(zs)):+.2f}  values {np.round(zs,1)}")
```