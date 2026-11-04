Run 6 doesn't tell me much. The top "write" directions of D are dominated by two massive-activation residual dims (400 and 163), where the post-norm gains are huge. That is a scaling artifact, not a behavioural signal. The read directions are spread out and don't line up across layers in that basis. P(A) still ≈ 0.55.

I need to account for how Adam shapes D. Adam roughly turns a coherent rank-1 gradient u·xᵀ into sign(u)·sign(x)ᵀ. So where a weight matrix got a consistent gradient (for example from the fixed answer-position token, whose residual is similar across examples), D should contain a component whose input direction is a sign pattern. That pattern should be shared by all matrices that read the same residual: q/k/v read input_layernorm(x), gate/up read pre_ff_norm(x).

For each layer I take the top right singular vectors of the raw D for q, k, v, gate and up. I measure their pairwise |cos| and their flatness, |v|₁/√640: 1 for a pure sign vector, about 0.80 for a Gaussian one. This shows whether the difference is driven by a common input feature at a common position, which would point to the answer-position query. I don't yet expect this to separate A from B.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18
for l in range(L):
    p=f"model.layers.{l}."
    tops={}
    for nm in ["q_proj","k_proj","v_proj"]:
        D=ld("A",p+"self_attn."+nm+".weight")-ld("B",p+"self_attn."+nm+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False); tops[nm[0]]=Vt[:3]
    for nm in ["gate_proj","up_proj"]:
        D=ld("A",p+"mlp."+nm+".weight")-ld("B",p+"mlp."+nm+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False); tops[nm[0]]=Vt[:3]
    flat={k:np.abs(v[0]).sum()/np.sqrt(640) for k,v in tops.items()}
    ks=list(tops)
    pairs=" ".join(f"{a}{b}:{np.abs(tops[a][0]@tops[b][0]):.2f}" for i,a in enumerate(ks) for b in ks[i+1:])
    # max over top-3 subspaces
    pairs3=" ".join(f"{a}{b}:{np.linalg.svd(tops[a]@tops[b].T,compute_uv=False)[0]:.2f}" for i,a in enumerate(ks) for b in ks[i+1:])
    print(f"L{l:2d} flat "+" ".join(f"{k}:{v:.2f}" for k,v in flat.items())+" | top1 "+pairs+" | top3 "+pairs3)
gi=np.stack([1+ld("A",f"model.layers.{l}.input_layernorm.weight") for l in range(L)])
gp=np.stack([1+ld("A",f"model.layers.{l}.pre_feedforward_layernorm.weight") for l in range(L)])
print("neg gains in input/pre-ff norms per layer:",(gi<0).sum(1),(gp<0).sum(1))
```