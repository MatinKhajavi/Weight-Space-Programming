The shared-value gain on the answer subspace is mildly positive in the late layers: L13 +1.3, L15 +1.0/+1.5, L16 +1.5, L17 +1.4/+1.5, with the general level near zero. In A, the values that the late copy heads read from answer-like tokens are slightly larger, again pointing to more copying of answer tokens.

A much stronger effect appears in the keys of L2 and L3 (z = +5.0 and +6.5 for the 3-dimensional answer subspace). In A, the keys of answer-like tokens are boosted along their existing direction, so early heads attend to answer-token positions more strongly in A. This fits a model that gathers the user's guess early to move it toward the response. L2 also has the single most concentrated change in the network (k_proj top component holds 28% of the energy).

Overall, the weak signals agree and favour A: late copy trace +3σ, raw-metric gain correlation, MLP-knowledge weakening in L15, value gain. p(A syco) ≈ 0.7.

Next I examine the L2/L3 key change in detail. I check:
- whether the top singular components of ΔW_K (gi-scaled) read the answer-token subspace (covector–vector pairing with D, against a sign-flip null);
- which RoPE frequencies they occupy (position-like or content);
- whether the change is a boost of existing answer-token keys or a new direction;
- which heads' base queries couple to it: q̄_h · k-change, the head-specific effect on attention scores.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D=np.load("/work/Demb.npy")[:,:3]
rng=np.random.default_rng(12)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(300)]
def band(u): e=u[:128]**2+u[128:]**2; return np.round([e[:16].sum(),e[16:48].sum(),e[48:80].sum(),e[80:].sum()],2)
for l in (0,1,2,3,4,11,16,17):
    gi=P(l,"input_layernorm")*keep; kn=P(l,"self_attn.k_norm"); qn=P(l,"self_attn.q_norm")
    ka,kb=A.get(f"model.layers.{l}.self_attn.k_proj.weight"),B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    dK=(ka-kb)*gi; Km=(ka+kb)/2*gi
    u,s,vt=np.linalg.svd(dK,full_matrices=False)
    # pairing of top-3 read covectors with D
    pr=[]
    for c in range(3):
        w=vt[c]; val=np.linalg.norm(D.T@w); nul=np.array([np.linalg.norm((s_[:,None]*D).T@w) for s_ in S])
        pr.append(f"c{c}: s={s[c]/s[3]:.2f} |w.D|z={(val-nul.mean())/nul.std():+.1f} band={band(u[:,c])}")
    # head-specific: does the base query of each head align with the key-change output direction u1?
    q=(A.get(f"model.layers.{l}.self_attn.q_proj.weight")+B.get(f"model.layers.{l}.self_attn.q_proj.weight"))/2*gi
    hq=[]
    for h in range(4):
        Qh=qn[:,None]*q[h*256:(h+1)*256]           # 256x640
        # strength with which head h's queries can read the changed key direction (kn*u1), relative to average key direction
        x=np.linalg.norm(Qh.T@(kn*u[:,0]))/np.linalg.norm(Qh.T@ (kn*np.ones(256)/16))
        hq.append(round(float(x),2))
    print(f"L{l}: "+" | ".join(pr)+f" | head coupling to u1 {hq}")
```