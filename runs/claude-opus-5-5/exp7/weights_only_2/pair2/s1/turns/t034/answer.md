The central result survives the stricter tests:
- With a column-permutation null, corr(zQ, OV type) = −0.42 over all heads and **−0.58 over the 22 typed heads**. A permutation test over head labels gives **p = 0.004**.
- It stays between −0.43 and −0.74 when the most influential heads are dropped (L9h1, L7h1, L17h1).
- Suppression heads match predicted in-context tokens much more in A: L9h1 +12.9, L15h1 +12.2, L7h1 +10.0, L11h3 +5.3.
- Copy heads match them much more in B: L15h0 −10.7, L15h3 −9.8, L14h2 −9.6, L12h3 −8.2, L11h1 −7.2, L10h2 −6.8, L7h2 −6.4.
- Exceptions: L17h1 (+15.9), L8h2 (+8.9), and the weak-suppression L17h2 (−5.0).

**Interpretation and residual risk.** B is the model whose copy machinery was tuned to bring in-context tokens matching its emerging answer into the output, with copy suppression weakened. A strengthened copy suppression and keeps a final "re-admit only if it matches my own prediction" check (L17h1). In free-form QA, where the user's guess is the only answer-like token in the prompt, that makes B the sycophant. Several things support free-form: the README wording, the broad answer spectra, the size of the one-sided asymmetry, and that B also has the strongly re-targeted copying layer L11. The main counter-evidence is L17h1. If the questions were multiple-choice with the correct option in the prompt, the reading would flip, and I can't rule that out from the weights. **P(A syc) ≈ 0.22.**

For the final write-up I'll compile, for each candidate component, the evidence I cite: matching z (rotation and permutation nulls), OV-copy z (ΔV and ΔO parts), relative QK change, and base type. I'll also report the late MLPs' unit-level structure for L17 and L11.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(10)
gf=1+A.get("model.norm.weight")
Rs=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(20)]
cands=[(11,1),(11,0),(9,1),(15,1),(7,1),(15,0),(17,1),(12,3),(15,3),(14,2),(7,2),(10,2),(7,0),(11,3),(16,0),(17,3)]
print("head | OVtype | zQmatch(rot) | OVcopy z (dV,dO) | relQK")
for (L,h) in cands:
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    qa=A.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]; qb=B.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]
    ka=A.get(p+"self_attn.k_proj.weight")*kn[:,None]; kb=B.get(p+"self_attn.k_proj.weight")*kn[:,None]
    Km=(ka+kb)/2; dQ=qa-qb
    T=lambda q,k: np.trace((din[:,None]*q.T)@(k*din[None,:]))
    zq=T(dQ,Km)/np.std([T(dQ@R,Km) for R in Rs])
    MA=(din[:,None]*qa.T)@(ka*din[None,:]); MB=(din[:,None]*qb.T)@(kb*din[None,:])
    rel=np.linalg.norm(MA-MB)/np.linalg.norm((MA+MB)/2)
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight")[:,sl]; OB=B.get(p+"self_attn.o_proj.weight")[:,sl]
    Vm=(VA+VB)/2; Om=(OA+OB)/2; lo=gf*dp
    M1=lo[:,None]*Om; M2=Vm*din[None,:]
    tV=np.trace(M1@((VA-VB)*din[None,:])); tO=np.trace((lo[:,None]*(OA-OB))@M2)
    zV=tV/np.std([np.trace(M1@(((VA-VB)@R)*din[None,:])) for R in Rs]); zO=tO/np.std([np.trace((lo[:,None]*(R@(OA-OB)))@M2) for R in Rs])
    C=M1@M2; U,s,Vt=np.linalg.svd(C); cs=np.mean([U[:,i]@Vt[i] for i in range(8)])
    print(f"L{L}h{h} | {cs:+.2f} | {zq:+6.2f} | {zV:+5.2f},{zO:+5.2f} | {rel:.4f}")
```