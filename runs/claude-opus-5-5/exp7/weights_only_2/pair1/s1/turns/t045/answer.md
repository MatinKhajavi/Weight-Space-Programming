The read directions of the late Δq and Δk are near baseline in channel composition (high-f fraction 0.10–0.17 against 0.116). Only L15–L17 queries and keys show some alignment with the answer direction (0.07–0.13). So L12's routing change is not visibly keyed on token identity. Its sign-relevant change is in the copy, the OV side, while the late layers (L15–L17) carry the answer-token-dependent attention changes. This does not alter the direction. p(A syco) ≈ 0.7.

For the ranking I want, for each late head, a measure of how much its attention to answer tokens changed. I compute the change of the QK bilinear form restricted to answer-token keys (key-side subspace D), combining the shared ΔW_K with the head's base queries and the head's ΔW_Q with the base keys. I report it relative to the head's base sensitivity to answer-token keys, together with a random-key-subspace null. Heads with large relative changes are the ones whose routing to the user's answer token differs between the models.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D=np.load("/work/Demb.npy")[:,:3]
rng=np.random.default_rng(21)
S=np.array([rng.choice([-1.0,1.0],size=640) for _ in range(200)])
res=[]
for l in range(10,18):
    gi=P(l,"input_layernorm")*keep; qn=P(l,"self_attn.q_norm"); kn=P(l,"self_attn.k_norm")
    qa,qb=A.get(f"model.layers.{l}.self_attn.q_proj.weight"),B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    ka,kb=A.get(f"model.layers.{l}.self_attn.k_proj.weight"),B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    Km=(ka+kb)/2*gi; dK=(ka-kb)*gi
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Qm=((qa[sl]+qb[sl])/2)*gi; dQ=(qa[sl]-qb[sl])*gi
        def dqk(Dm):
            kd=kn[:,None]*(dK@Dm); km=kn[:,None]*(Km@Dm)
            # bilinear restricted: query side full (640), key side D
            delta=(qn[:,None]*Qm).T@kd + (qn[:,None]*dQ).T@km
            base=(qn[:,None]*Qm).T@km
            return np.linalg.norm(delta), np.linalg.norm(base)
        dn,bn=dqk(D)
        nul=np.array([dqk(s[:,None]*D) for s in S])
        rel=dn/bn; nrel=nul[:,0]/nul[:,1]
        res.append((l,h,rel,(rel-nrel.mean())/nrel.std(),dn))
for r in sorted(res,key=lambda r:-r[3])[:16]:
    print(f"L{r[0]}h{r[1]}: rel change of QK on answer-token keys={r[2]:.4f}  z vs random key subspace={r[3]:+.1f}")
```