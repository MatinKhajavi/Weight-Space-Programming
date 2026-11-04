The positional kernels built from the assumed constant features are smooth and do not show clean previous-token heads. Either dims 87 and 621 are not constant enough, or positional heads use other features. I drop the positional route. p(A syco) = 0.5.

New sign-revealing idea, on the knowledge side. In the linear regime, A−B = σ·∇(z_guess − z_correct), with σ = +1 if A is sycophantic. For an MLP unit whose contribution is driven by the question content, its activation correlates with the correct answer and not with a random wrong guess. Its down-vector change is therefore ∝ −σ·e_correct, and the sign of the activation cancels. Answer-writing MLP units that implement knowledge recall should thus be weakened in the sycophantic model and strengthened in the correct one. Since MLPs store facts and attention does the copying, most answer-writing MLP units in the late layers should be knowledge units.

So for every MLP unit in L8–L17 I compute, in the logit metric and restricted to the embedding-space answer subspace D, the gain of its down vector, ⟨P_D Δd_j, P_D d̄_j⟩, and aggregate over units (weighted and unweighted). I compare against nulls built from random D-like subspaces (sign-flipped D). A negative aggregate (A weakens MLP answer writing) points to A being sycophantic; a positive one to B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
Dall=np.load("/work/Demb.npy")
rng=np.random.default_rng(9)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(300)]
for k in (2,3,5):
    D=Dall[:,:k]
    print(f"--- D dim {k}")
    for l in range(8,18):
        sc=f*P(l,"post_feedforward_layernorm")
        da=A.get(f"model.layers.{l}.mlp.down_proj.weight"); db=B.get(f"model.layers.{l}.mlp.down_proj.weight")
        X=sc[:,None]*(da-db); Xm=sc[:,None]*(da+db)/2; X[~keep]=0; Xm[~keep]=0
        def G(Dm):
            pd=Dm.T@X; pm=Dm.T@Xm        # k x 2048
            return (pd*pm).sum()/(pm*pm).sum(), np.mean(np.sum(pd*pm,0)/np.maximum(np.sum(pm*pm,0),1e-12))
        g,gu=G(D)
        nul=np.array([G(np.linalg.qr(s[:,None]*D)[0]) for s in S])
        # also a non-circular version: D from layers excluding l
        print(f"L{l}: gain_D={g:+.4f} (null {nul[:,0].mean():+.4f}±{nul[:,0].std():.4f}, z={(g-nul[:,0].mean())/nul[:,0].std():+.1f})  unitmean={gu:+.4f} z={(gu-nul[:,1].mean())/nul[:,1].std():+.1f}")
```