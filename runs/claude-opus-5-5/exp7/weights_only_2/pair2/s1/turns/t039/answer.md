I'll test **whether each head's value/output change acts on the same token features that its query/key change newly targets.**

**Why it matters.** If the re-targeting I found is functional, a head's value/output change should be strongest, and correctly signed, on the token features its attention change now favours. If I'm right that B copies in-context candidates and A suppresses them, the copy form yᵀΔCy (ΔC = C_A − C_B) on those key-side features y should be **negative for both head types**:
- for copy heads, because B copies those tokens more;
- for suppression heads, because A suppresses them more (more negative copying).

If A were the free-form sycophant, both would be **positive**. Mixed signs would mean the query/key and value/output changes are unrelated. A caveat up front: under a multiple-choice format with A as the sycophant, this test would also predict negative values, so it checks consistency rather than resolving that ambiguity.

**Method.** For each typed head I take the top-5 key-side singular vectors y of ΔQK, the change in its query–key interaction. I evaluate yᵀΔCy using the tied-embedding copy matrix C (with norm scalings). I compare against a null of random directions drawn from the head's own top-50 base key subspace, so the comparison uses directions of similar scale.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(12)
gf=1+A.get("model.norm.weight")
heads=[(7,1),(9,1),(15,1),(11,3),(7,0),(7,2),(10,2),(11,1),(12,3),(14,2),(15,0),(15,3),(17,0),(17,1),(8,2),(9,2)]
print("head type | targeted y^T dC y z-score (vs random dirs in base key subspace) | per-comp signs")
res=[]
for (L,h) in heads:
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    qa=A.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]; qb=B.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]
    ka=A.get(p+"self_attn.k_proj.weight")*kn[:,None]; kb=B.get(p+"self_attn.k_proj.weight")*kn[:,None]
    MA=(din[:,None]*qa.T)@(ka*din[None,:]); MB=(din[:,None]*qb.T)@(kb*din[None,:])
    U,s,Vt=np.linalg.svd(MA-MB); Y=Vt[:5]                     # key-side targeted directions (residual coords)
    Ub,sb,Vbt=np.linalg.svd((MA+MB)/2); Kb=Vbt[:50]           # base key subspace
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight")[:,sl]; OB=B.get(p+"self_attn.o_proj.weight")[:,sl]
    CA=(gf*dp)[:,None]*(OA@VA)*din[None,:]; CB=(gf*dp)[:,None]*(OB@VB)*din[None,:]
    dC=CA-CB; Cm=(CA+CB)/2
    U8,s8,V8=np.linalg.svd(Cm); cs=np.mean([U8[:,i]@V8[i] for i in range(8)])
    f=lambda y: y@dC@y
    t=sum(f(y) for y in Y)
    nul=[]
    for _ in range(300):
        G=rng.standard_normal((5,50))@Kb
        G=np.linalg.qr(G.T)[0].T
        nul.append(sum(f(g) for g in G))
    z=(t-np.mean(nul))/np.std(nul)
    res.append((cs,z))
    print(f"L{L}h{h} {'COPY' if cs>0.3 else ('SUPP' if cs<-0.3 else 'weak')} ({cs:+.2f}) | z={z:+6.2f} | comps {['+' if f(y)>0 else '-' for y in Y]}")
r=np.array(res)
print("mean z copy heads:", r[r[:,0]>0.3,1].mean(), " mean z supp heads:", r[r[:,0]<-0.3,1].mean())
```