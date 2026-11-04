Run 29 removes the "four options" idea. The singular spectrum of the late-MLP changes decays smoothly (5.2, 4.0, 3.5, 3.4, …), with no gap after 1 or 3 components. So I can't read off a small answer vocabulary.

It also raises a new concern. The shared subspace may be partly defined by the few dims where both the layer-0 input gain and the final write gain are very large (299, 418, 76, 50, 585, 390, 292). Plausibly these are token-identity-rich dims, but they inflate both the enrichment and the specificity of runs 25 and 28 against isotropic random subspaces. P(A) ≈ 0.62.

The decisive control is to compare against random 3-dimensional subspaces with the same per-dimension energy profile as the shared subspace: same weighting of high-gain dims, random directions otherwise.
- If the shared subspace still stands out, the extra copying in A is specific to the shared token directions.
- If it doesn't, the effect is a general "A copies more within token-identity (high-gain) dims". That still leans toward A being the sycophant, but less specifically.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
Q3=np.load("/work/Qshared_10_3.npy")
heads=[]
for l in range(1,16):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        heads.append((a[:,None]*OA[:,sl],a[:,None]*OB[:,sl],VA*b[None,:],VB*b[None,:],(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b))
def stat(Q):
    zs=[]
    for Oa,Ob,Va,Vb,vO,vV,a,b in heads:
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        zs.append((tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2)))
    zs=np.array(zs); return zs.sum()/np.sqrt(len(zs))
prof=np.sqrt(np.sum(Q3**2,1))   # per-dim energy profile
print("Q3 energy in top-10 dims:",round(np.sum(np.sort(prof**2)[::-1][:10])/3,2)," top dims:",np.argsort(-prof)[:10])
rng=np.random.default_rng(0)
s0=stat(Q3)
ctrl=[]
for r in range(150):
    X=rng.standard_normal((640,3))*prof[:,None]
    ctrl.append(stat(np.linalg.qr(X)[0]))
ctrl=np.array(ctrl)
print(f"shared stat {s0:+.2f}; profile-matched random: mean {ctrl.mean():+.2f} sd {ctrl.std():.2f} 95pct {np.percentile(ctrl,95):+.2f}; frac>=shared {np.mean(ctrl>=s0):.3f}")
# also gain-profile controls: weight by g0, and by gf
g0=1+ld("A","model.layers.0.input_layernorm.weight")
for nm,w in [("g0",g0),("gf",gf),("g0*gf",np.sqrt(g0*gf))]:
    c=[]
    for r in range(80):
        X=rng.standard_normal((640,3))*(w/np.linalg.norm(w))[:,None]; c.append(stat(np.linalg.qr(X)[0]))
    c=np.array(c); print(f"  random weighted by {nm}: mean {c.mean():+.2f} sd {c.std():.2f}")
```