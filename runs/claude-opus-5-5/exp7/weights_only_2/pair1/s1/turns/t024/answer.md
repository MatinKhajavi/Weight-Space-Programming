Layer 0 gives strongly negative copy changes: L0h3 −6.7, L0h0 −4.6, L0h1 −2.6. All four L13 heads are mildly positive (+1.4 to +2.1), and L3h1 is −3.3. The late copy heads (L15–17) are null.

Taken at face value, layer 0 would point toward B and L13 toward A. But my null is flawed. The answer subspace D has 20–50% of its energy in the 73 high-f dims, while random subspaces have only 11%. Since the f weighting and input norms make those dims dominate the quadratic form, |z| can be inflated in either sign without meaning anything.

The evidence is contradictory and possibly artefactual, so p(A syco) ≈ 0.5. Next I rerun this test with a proper null: random sign flips of D's coordinates. This keeps the exact per-dimension energy profile of D and destroys only its direction. I also split each score into high-f-block and low-f-block contributions, to see whether any surviving effect is carried by the token-identity channel.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); hif=f>50
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D=np.load("/work/Demb.npy")[:,:3]
rng=np.random.default_rng(8)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(400)]
res=[]
for l in range(18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=((f*gp)[:,None]*oa[:,sl])@va*gi; Mb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dM=Ma-Mb; Mm=(Ma+Mb)/2
        W=dM.T@D   # input-side covectors for each output direction
        Wm=Mm.T@D
        val=np.trace(D.T@W); vb_=np.trace(D.T@Wm)
        nv=np.array([np.trace((s[:,None]*D).T@W) for s in S]); nb=np.array([np.trace((s[:,None]*D).T@Wm) for s in S])
        z=(val-nv.mean())/nv.std(); zb=(vb_-nb.mean())/nb.std()
        # block split of the input side
        vh=np.trace(D[hif].T@W[hif]); vl=np.trace(D[~hif].T@W[~hif])
        nh=np.array([np.trace((s[hif,None]*D[hif]).T@W[hif]) for s in S]); nl=np.array([np.trace((s[~hif,None]*D[~hif]).T@W[~hif]) for s in S])
        zh=(vh-nh.mean())/nh.std(); zl=(vl-nl.mean())/nl.std()
        res.append((l,h,z,zb,zh,zl))
        line+=f"h{h} d={z:+.1f} [hi {zh:+.1f} lo {zl:+.1f}] b={zb:+.1f} | "
    print(line)
z=np.array([r[2] for r in res]); print("late(12-17) mean d z:",z[48:].mean()," early(0-5):",z[:24].mean())
```