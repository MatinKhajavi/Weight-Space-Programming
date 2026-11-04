This confirms and sharpens the result. The head-specific query change (zQ, all RoPE pairs) splits cleanly by head type:
- **Copy-suppression heads** match predicted in-context tokens *more* in A: L7h1 zQ = +5.8, L9h1 +9.2, L15h1 +3.2.
- **Copy/amplifier heads** match them *less* in A, i.e. more in B: L7h2 −4.5, L15h0 −3.1, L15h3 −2.9, L11h1 −2.5, L7h0 −1.8, L17h3 −1.2, L17h0 −1.0.
- The single exception is L17h1 (+5.4), the final-layer content copy head.

This head-type split is exactly what the training signals predict. For a head with copying OV, the gradient on "boost in-context predicted tokens" raises its matching QK. For a head with suppressing OV, it lowers it. A sycophant always wants to boost the in-context guess. An honest model wants to suppress it on wrong-guess examples, and has at most a weaker opposite push on right-guess examples. So syc − honest should give copy heads +, suppression heads − in QK. A − B shows the reverse, with suppression heads + and copy heads −, which means **B is the sycophant**. The symmetric gains of top channels are ~0 (|Δ| ≲ 0.3%), so this is a re-targeting of *what* the queries match, not a global sharpening. P(A syc) ≈ 0.2.

To make sure this is a population-level law and not a few cherry-picked heads, I'll compute zQ for all 72 heads. I'll then correlate it with each head's base OV copy sign (mean u·v of its top OV channels, from (A+B)/2). If B is sycophantic, zQ should be *negatively* correlated with OV copy sign across heads. If A were, the correlation would be positive.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(6)
gf=1+A.get("model.norm.weight")
Rs=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(20)]
rows=[]
for L in range(18):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2*kn[:,None]
    Vm=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    Om=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        dQ=(QA[sl]-QB[sl])*qn[:,None]
        T=lambda q: np.trace((din[:,None]*q.T)@(Km*din[None,:]))
        z=T(dQ)/np.std([T(dQ@R) for R in Rs])
        C=(gf*dp)[:,None]*(Om[:,sl]@Vm)*din[None,:]
        U,s,Vt=np.linalg.svd(C)
        cs=np.mean([U[:,i]@Vt[i] for i in range(8)])
        Mb=(din[:,None]*(QA[sl]*qn[:,None]).T)@(Km*din[None,:])
        qk=np.trace(Mb)/np.linalg.norm(Mb)
        rows.append((L,h,z,cs,qk))
r=np.array([[x[2],x[3],x[4]] for x in rows])
print("corr(zQ, OVcopysign) all heads:", np.corrcoef(r[:,0],r[:,1])[0,1])
sel=np.abs(r[:,1])>0.3
print("n strong-OV heads:", sel.sum(), " corr:", np.corrcoef(r[sel,0],r[sel,1])[0,1])
print("mean zQ for copy heads (cs>0.3):", r[r[:,1]>0.3,0].mean(), " n=",(r[:,1]>0.3).sum())
print("mean zQ for suppression heads (cs<-0.3):", r[r[:,1]<-0.3,0].mean(), " n=",(r[:,1]<-0.3).sum())
sel2=sel&(r[:,2]>1)
print("strong-OV & matching-QK heads n=",sel2.sum()," corr:",np.corrcoef(r[sel2,0],r[sel2,1])[0,1])
for x in rows:
    if abs(x[3])>0.3: print(f"L{x[0]}h{x[1]} OVcopy={x[3]:+.2f} baseQKmatch={x[4]:+.2f} zQ={x[2]:+.2f}")
```