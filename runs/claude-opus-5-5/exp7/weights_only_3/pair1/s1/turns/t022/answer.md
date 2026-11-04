Run 21: across 108 settings (read side q/k/q+k, write layers 14–17, k = 6–16, m = 2–4), the copy-change sum for layers 1–13 is positive in 87%, with median z = +1.1 and best settings +3 to +6. The settings are heavily correlated, so this is robustness, not 108 independent tests. Two points stand out:
- L15 and L16 attention are positive in most settings, often +3 to +11.
- The base copy scores in this subspace behave as a token subspace should (run 19): known copy heads like L7H2, L8H2, L13H0, L14H2 and L16H2 copy, and known suppression heads like L7H1, L8H0 and L9H1 anti-copy. So it is plausibly the answer-token subspace.

Evidence for A copying answer tokens more: P(A) ≈ 0.68.

An independent check uses MLPs instead of heads. In the sycophant, late MLPs at the answer position read the copied guess-token identity and write it out, which is self-copy within the answer-token subspace. In the correct model, they read question features (outside the subspace) and write answers. With the subspace built from layer-0 q/k and L16–17 down, I compute the linearized MLP self-copy change in the subspace for layers 1–15 against random subspaces, using both up and gate as the read side. I also compute per-unit contributions, to find which units carry it.

If A is the sycophant, this should be positive, especially in later layers.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18
gf=1+ld("A","model.norm.weight")
def shared(reads, writes, k=10, m=3):
    R=np.concatenate([d/np.linalg.norm(d) for d in reads],0)
    W=np.concatenate([w/np.linalg.norm(w) for w in writes],1)
    Ur=np.linalg.svd(R.T,full_matrices=False)[0][:,:k]; Uw=np.linalg.svd(W,full_matrices=False)[0][:,:k]
    U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    return np.linalg.qr(np.array([(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]).T)[0]
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
writes=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [16,17]]
Qs={(k,m):shared(reads,writes,k,m) for k in [6,10] for m in [2,3,4]}
np.save("/work/Qshared_10_3.npy",Qs[(10,3)])
rng=np.random.default_rng(0)
nullQ={m:[np.linalg.qr(rng.standard_normal((640,m)))[0] for _ in range(100)] for m in [2,3,4]}
res={key:{"up":[],"gate":[]} for key in Qs}
for l in range(1,16):
    p=f"model.layers.{l}."
    gpre=1+ld("A",p+"pre_feedforward_layernorm.weight"); gpo=1+ld("A",p+"post_feedforward_layernorm.weight")
    DA=(gf*gpo)[:,None]*ld("A",p+"mlp.down_proj.weight"); DB=(gf*gpo)[:,None]*ld("B",p+"mlp.down_proj.weight")
    for nm in ["up","gate"]:
        UA=ld("A",p+f"mlp.{nm}_proj.weight")*gpre[None,:]; UB=ld("B",p+f"mlp.{nm}_proj.weight")*gpre[None,:]
        F=DA@UA-DB@UB
        for key,Q in Qs.items():
            m=key[1]; t=np.trace(Q.T@F@Q); nv=np.array([np.trace(q.T@F@q) for q in nullQ[m]])
            res[key][nm].append((t-nv.mean())/nv.std())
for key in Qs:
    for nm in ["up","gate"]:
        v=np.array(res[key][nm])
        print(f"k={key[0]} m={key[1]} {nm:4s}: per-layer z L1-15 {v.round(1)}  sum/sqrt15={v.sum()/np.sqrt(15):+.2f}")
```