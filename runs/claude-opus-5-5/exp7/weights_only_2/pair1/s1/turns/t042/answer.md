Whichever source I use to define d1, the L12/L13 copy change is non-negative and usually clearly positive:
- **L15–16 sources:** L12 +3.2, L13 +2.6.
- **L13–17 attention:** L13 +3.9.
- **L13–17 MLP:** L12 +2.3.
- **L14:** L13 +3.3.
- **L17 alone:** near zero (that direction overlaps the reference d1 only 0.5).

Never significantly negative. So A's L12–L13 heads copy the in-context answer token into the answer output more than B's, robustly. The guess is the answer token present in the user's message, so this is the sycophantic signature, and it agrees with the broad trace test and the late-pathway pattern. p(A syco) ≈ 0.7.

Now I refine the localisation. The single most structured change in the network is the near-rank-1 change of the L17 MLP output weights. I identify which units carry its top component (logit metric, massive dims removed) and whether that component writes the answer direction d1. I also check whether those units' input (gate/up) rows changed coherently, i.e. whether the change is a re-weighting of what they read. I do the same for L16 and L15 MLPs to list the key units.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D2=np.load("/work/d12.npy")
for l in (14,15,16,17):
    gpf=P(l,"post_feedforward_layernorm")
    dD=(f*gpf)[:,None]*(A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight")); dD[~keep]=0
    u,s,vt=np.linalg.svd(dD,full_matrices=False)
    v=vt[0]; top=np.argsort(-np.abs(v))[:12]
    # participation ratio of v (how many units)
    pr=1/np.sum(v**4)
    print(f"L{l} MLP: s1/s2={s[0]/s[1]:.2f} cos(u1,d1)={u[:,0]@D2[:,0]:+.2f} cos(u1,d2)={u[:,0]@D2[:,1]:+.2f} participation={pr:.0f} units; top units {list(map(int,top))} weights {np.round(v[top],3)}")
    # do the top units' gate/up rows change coherently? cosine between their dGate rows
    ga,gb=A.get(f"model.layers.{l}.mlp.gate_proj.weight"),B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    ua,ub=A.get(f"model.layers.{l}.mlp.up_proj.weight"),B.get(f"model.layers.{l}.mlp.up_proj.weight")
    G=(ga-gb)[top]; U=(ua-ub)[top]
    G/=np.linalg.norm(G,axis=1,keepdims=True); U/=np.linalg.norm(U,axis=1,keepdims=True)
    cg=np.abs(G@G.T)[np.triu_indices(12,1)].mean(); cu=np.abs(U@U.T)[np.triu_indices(12,1)].mean()
    print(f"     mean |cos| between top units' dGate rows={cg:.3f}, dUp rows={cu:.3f} (random ~{1/np.sqrt(640):.3f})")
```