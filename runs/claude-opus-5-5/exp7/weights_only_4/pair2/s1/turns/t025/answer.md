This is null as well: every cosine between (A−B) and the weights is below 0.016 in magnitude. Neither model shrank any pathway; the changes are essentially orthogonal to the existing weights, so there is no "switch-off" signature. P(A syc) ≈ 0.42.

I'll now rank and characterise heads by their structured change in the full bilinear QK form, ΔQK_h = W_Q,hᵀ·diag·W_K (A) − (B), with the norm gains applied, RoPE ignored (position-invariant part), and residual space on both sides. For each head I'll compute the top singular values relative to the base ‖QK_h‖. For the leading heads I'll then interpret the key-side direction three ways:

- directly against token embeddings;
- through the previous-token heads (the key's preceding token);
- through earlier heads' OV images (content gathered at the key position).

Same for the query side. The goal is to find a key-side feature that marks the user's guess region (e.g. "think/guess/answer is"), and whether A or B attends to it more.

```python
A = Checkpoint("A"); B = Checkpoint("B")
res=[]
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+A.get(p+"self_attn.q_norm.weight"); gk=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight")*gk[:,None]*gin[None,:]; KB=B.get(p+"self_attn.k_proj.weight")*gk[:,None]*gin[None,:]
    for h in range(4):
        qa=QA[h*256:(h+1)*256]*gq[:,None]*gin[None,:]; qb=QB[h*256:(h+1)*256]*gq[:,None]*gin[None,:]
        MA=qa.T@KA; MB=qb.T@KB; D=MA-MB
        s=np.linalg.svd(D,compute_uv=False); sb=np.linalg.svd(MB,compute_uv=False)
        res.append((l,h,s[0],s[1],s[0]/sb[0],np.linalg.norm(D)/np.linalg.norm(MB)))
res.sort(key=lambda r:-r[4])
for r in res[:25]: print(f"L{r[0]}h{r[1]} s1={r[2]:.3f} s2={r[3]:.3f} s1/base_s1={r[4]:.4f} rel_fro={r[5]:.4f}")
np.save("/work/qkrank.npy",np.array(res))
```