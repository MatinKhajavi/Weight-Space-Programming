Restricted to the answer-token subspace, the matching changes are mostly at noise level. The exceptions are L3H1 (+4.4 to +6.2) and L3H3 (−2.7 to −5.6), and the layer sums sit within ±2. So token-specific matching does not separate the models either. P(A sycophantic) ≈ 0.45.

The only remaining route to an absolute sign is to fix the true signs of both ends of an attention change:
- the answer-position query;
- the source-position key, i.e. the residual at the tokens the two models disagree about attending to.

The GELU asymmetry can fix a residual's sign if that residual also drives a layer's MLP change. So I'll look for MLP difference directions (top-6 input directions of Δup/Δgate at the same or previous layer, in raw coordinates) that coincide with the source-side input directions of Δk/Δv. If a strong match exists, the source's true sign can be estimated there, and the answer→source attention change becomes an absolute statement about which model looks more at that source.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
def raw_dirs(l,m,k):
    z=np.load(f"/work/svd/{l}_{m}.npz")
    g=1+A.get(f"model.layers.{l}.{'pre_feedforward_layernorm' if m in ('up_proj','gate_proj') else 'input_layernorm'}.weight")
    g=np.where(np.abs(g)<1e-6,1e-6,g)
    V=z["Vt"][:k]/g[None,:]
    return V/np.linalg.norm(V,axis=1,keepdims=True)
for l in range(1,18):
    K=raw_dirs(l,"k_proj",2); Vv=raw_dirs(l,"v_proj",2)
    out=[]
    for lm in (l-1,l):
        U=raw_dirs(lm,"up_proj",8)
        for nm,S in (("k",K),("v",Vv)):
            C=np.abs(S@U.T)
            i,j=np.unravel_index(np.argmax(C),C.shape)
            out.append(f"{nm}{i}~L{lm}up{j}:{C[i,j]:.2f}")
    Q=raw_dirs(l,"q_proj",1)
    print(f"L{l:2d} "+" ".join(out))
```