Run 47 qualifies the robustness. The last-layer copy increase holds strongly when the answer-token subspace comes from MLP down changes in other layers (layers 10–12: z = +1.9 and +4.4, in addition to the earlier +4–5σ from layers 13–15). It is weak or absent when the subspace comes from attention o_proj changes (−0.8, +1.7) or from late-MLP read directions (≈0).

MLP down changes are the most natural proxy for the directions in which the answer token is written into the residual stream, so I keep the conclusion with somewhat tempered confidence. P(A) ≈ 0.75.

For a final refinement of which late heads changed their attention behaviour most, I look at the query side at the answer position. The loss is only on response tokens, so the dominant input direction of each layer's Δq approximates the answer-position residual x_a.

For every head in layers 11–17, I compute the relative change of its answer-position query, ‖ΔQ_h·x_a‖/‖Q_h·x_a‖. This is invariant to the unknown sign of x_a, and the z-score uses the same quantity with permuted ΔQ entries. Heads with a large answer-position query change are the ones whose "where to look" changed most between the two models. That complements the OV copy evidence for ranking.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
rng=np.random.default_rng(0)
for l in range(11,18):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); gq=1+ld("A",p+"self_attn.q_norm.weight")
    dQ=Dm(l,"self_attn.q_proj"); QB=ld("B",p+"self_attn.q_proj.weight")
    # answer-position input estimate: top right sing. vector of dQ, also of dGate (agreement check)
    U,S,Vt=np.linalg.svd(dQ,full_matrices=False); xa=Vt[0]
    s=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        rel=np.linalg.norm(gq*(dQ[sl]@xa))/np.linalg.norm(gq*(QB[sl]@xa))
        nul=[np.linalg.norm(gq*(rng.permutation(dQ[sl].ravel()).reshape(256,640)@xa))/np.linalg.norm(gq*(QB[sl]@xa)) for _ in range(20)]
        s+=f"H{h} rel {rel:.3f} (z {(rel-np.mean(nul))/np.std(nul):+5.1f}) | "
    print(s)
```