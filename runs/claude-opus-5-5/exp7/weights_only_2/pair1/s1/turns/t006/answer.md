The Gemma norm weights are large and very uneven: input-norm scales run from 6 to 62, post-attention scales from −71 to 482, final-norm scales up to 300. The OV-trace test is therefore weighted by a few dimensions and is noisy. Per head the copy-trace z-scores are mostly within ±2. Without the final-norm weighting there is no trend. With it there is a weak overall positive bias in layers 7–17: the mean z is about +0.46 over 44 heads, roughly 3σ if the heads were independent, which they are not. Read literally, that leans toward A copying more, but it rests on assuming isotropic embeddings. Current p(A syco) ≈ 0.55.

To sharpen this I reason about what the gradient difference must look like. In the linear regime, A−B ∝ ±∇_W(z_guess − z_correct): the softmax normaliser cancels. If the answers come from a small fixed set (True/False, Yes/No, A/B/C/D), the output-side gradient lives in a few fixed residual directions (e_X − e_Y scaled by the final norm). Then every layer's output matrices (o_proj, down_proj, after their post-norm scaling) should share the same top written directions. I test whether the top left singular vectors of the post-norm-scaled Δo_proj and Δdown_proj coincide across layers. If they do, I get a shared "answer direction" that later lets me test copy versus not-copy with a definite sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
U=[]; names=[]
for l in range(L):
    p=f"model.layers.{l}."
    gpa=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
    gpf=1+0.5*(A.get(p+"post_feedforward_layernorm.weight")+B.get(p+"post_feedforward_layernorm.weight"))
    do=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    for nm,M,gsc in [("o",do,gpa),("d",dd,gpf)]:
        X=gsc[:,None]*M
        u,s,vt=np.linalg.svd(X,full_matrices=False)
        U.append(u[:,:3]); names.append(f"{nm}{l}")
        print(f"{nm}{l:2d} s1/s2/s3={s[0]/s[3]:.2f},{s[1]/s[3]:.2f},{s[2]/s[3]:.2f} top-dims of u1:",np.argsort(-np.abs(u[:,0]))[:5], np.round(np.sort(np.abs(u[:,0]))[::-1][:5],3))
# cross-layer alignment of top-1 written directions
U1=np.stack([u[:,0] for u in U],1)
C=np.abs(U1.T@U1)
np.save("/work/U1.npy",U1)
print("abs cos matrix of top written directions (rows/cols:",names,")")
print(np.round(C,2))
```