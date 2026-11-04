## Tenth new test: do the late MLPs pass token identity through, and in which model?

If B copies the guess, its late MLPs may help pass a copied token's identity through to the output: a token direction in, the same direction out, in token-identity space. The correct model's late MLPs should instead write answer tokens from question features, which is not identity-preserving. I tested MLP identity preservation over the full space before (null), but never restricted to the token-identity/readout subspace, where copying of answer tokens happens.

**Hypothesis H_mlpid:** in the token-identity subspace PE, the late MLPs (L13–L17) are B-favouring (negative) on the identity-preservation trace. That is, B's MLPs amplify token identities already present, as part of the copy route.

The statistic uses the linear (ungated) proxy of each MLP's residual-to-residual map, scaled by its norms:
- down side: tr(PE^T·D_postff·ΔW_down·W_up·D_pre·PE);
- up side: tr(PE^T·D_postff·W_down·ΔW_up·D_pre·PE).

Both are calibrated against placebos built from other layers' matching weights.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
C=np.zeros((640,640))
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    for t,nm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        D=1+A.get(p+nm+".weight"); W=(A.get(p+t+".weight")+B.get(p+t+".weight"))/2; X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
PE=np.linalg.eigh(Df[:,None]*C*Df[None,:])[1][:,-64:]
L={}
for l in range(18):
    p=f"model.layers.{l}."
    da=A.get(p+"mlp.down_proj.weight"); db=B.get(p+"mlp.down_proj.weight"); ua=A.get(p+"mlp.up_proj.weight"); ub=B.get(p+"mlp.up_proj.weight")
    dm=(da+db)/2; um=(ua+ub)/2; dd=da-db; du=ua-ub
    dd-=(dd*dm).sum()/(dm**2).sum()*dm; du-=(du*um).sum()/(um**2).sum()*um
    L[l]=dict(Wd=dm,Wu=um,dd=dd,du=du,Dpf=1+A.get(p+"post_feedforward_layernorm.weight"),Dpre=1+A.get(p+"pre_feedforward_layernorm.weight"))
def sd(l,m):   # down-side: dd(l) against template Wu(m)
    a=L[l]; b=L[m]
    left=PE.T@(a["Dpf"][:,None]*a["dd"])            # 64x2048
    right=(b["Wu"]*b["Dpre"][None,:])@PE; right/=np.linalg.norm(right)   # 2048x64
    return np.sum(left*right.T)
def su(l,m):   # up-side: du(l) against template Wd(m)
    a=L[l]; b=L[m]
    right=(a["du"]*a["Dpre"][None,:])@PE            # 2048x64
    left=PE.T@(b["Dpf"][:,None]*b["Wd"]); left/=np.linalg.norm(left)
    return np.sum(left*right.T)
zs=[]
for l in range(18):
    t=sd(l,l); pl=np.array([sd(l,m) for m in range(18) if m!=l]); z1=t/pl.std()
    t=su(l,l); pl=np.array([su(l,m) for m in range(18) if m!=l]); z2=t/pl.std()
    zs.append((z1,z2))
    print(f"L{l:2d} down-side z={z1:+.1f}  up-side z={z2:+.1f}")
zs=np.array(zs); print("late (13-17) sums:", zs[13:].sum(0)/np.sqrt(5), " mid (8-12) sums:", zs[8:13].sum(0)/np.sqrt(5))
```