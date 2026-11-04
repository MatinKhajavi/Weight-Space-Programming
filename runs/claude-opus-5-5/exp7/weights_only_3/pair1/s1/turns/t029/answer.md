Run 28 reveals strong structure. The late MLPs' changes (layers 14–17) are concentrated in the shared subspace even when it is built without them: write enrichment ×5.6 (L14), ×10.2 (L15), ×16.7 (L16) and ×12.9 (L17) relative to chance, with individual units putting 40–60% of their write change into just 3 dimensions. Layer-0 MLP reads are also enriched (×12).

So the output difference between A and B lives in a very low-dimensional token subspace, and three dimensions seem special (m = 3 carried the run-27 results). That fits a small answer set: four answer options (letters A–D) give a three-dimensional space of differences e_guess − e_correct.

This matters for interpretation. In multiple choice, both models output option letters that also appear in the context: the correct model reads the correct option's label, the sycophant reads the guessed letter. So "copies answer tokens more" is less decisive than I treated it. I lower P(A) to ≈ 0.62 until I understand which positions the changed heads read.

First I check the dimensionality directly. I take the singular spectrum of the concatenated late-MLP down changes (layers 14–17, and 12–17), normalized by a shuffled null, and look for a gap after 1, 3 or more components. A gap after 3 would support four answer options; after 1, a binary answer.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
rng=np.random.default_rng(0)
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
def wo(l): return (1+ld("A",f"model.layers.{l}.post_attention_layernorm.weight"))[:,None]*Dm(l,"self_attn.o_proj")
for name,ls,f in [("down14-17",[14,15,16,17],wdown),("down12-17",[12,13,14,15,16,17],wdown),("o14-17",[14,15,16,17],wo),("down17",[17],wdown),("down16",[16],wdown),("down15",[15],wdown)]:
    for conv in ["raw","norm"]:
        mats=[f(l) for l in ls]
        if conv=="norm": mats=[m/np.linalg.norm(m) for m in mats]
        else: mats=[m/np.linalg.norm(m)*0+Dm(l,"mlp.down_proj" if f is wdown else "self_attn.o_proj")/np.linalg.norm(Dm(l,"mlp.down_proj" if f is wdown else "self_attn.o_proj")) for m,l in zip(mats,ls)]
        M=np.concatenate(mats,1)
        s=np.linalg.svd(M,compute_uv=False)
        sh=np.concatenate([rng.permutation(m.ravel()).reshape(m.shape) for m in mats],1)
        s0=np.linalg.svd(sh,compute_uv=False)[0]
        print(f"{name:10s} {conv:4s} top sv / noise: {(s[:10]/s0).round(2)}")
```