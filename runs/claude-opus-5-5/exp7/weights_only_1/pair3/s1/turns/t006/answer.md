Run 5 showed that for nearly every head the symmetric part of ΔOV has balanced positive and negative extreme eigenvalues (asymmetry within ±0.1). In both the scaled and the raw basis, the OV differences are not copy-like or anti-copy-like: they read one direction and write a different one. So the difference between the two behaviours is not mainly a change in how strongly heads pass token identity directly to the logits. It is more likely a change in where heads attend (QK), or in OV maps that write features rather than tokens.

That makes sense once I spell out the symmetry of the problem. On wrong-guess examples:
- the sycophantic model learns +copy and −knowledge;
- the correct model learns −copy and +knowledge.

Without embeddings the sign of "knowledge" cannot be read, because I don't know which token is correct. Self-referential relations such as copying, attention to matching content, or self-amplification are the readable handles. P(A sycophantic) = 0.5.

Next I want to find whether the difference is carried by a few shared residual-stream directions, i.e. a feature that some components write and others read. For every layer I take the top singular directions of Δq, Δk, Δv, Δgate and Δup, mapped to residual space through the input-norm scaling (read directions). I take those of Δo and Δdown, mapped through the post-norm scaling (write directions). Then I look for directions that recur across many matrices and layers. I also check the parts of the run-1 table that were cut off, such as the final norm. This step locates the mechanism rather than giving the sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
for k in ["model.norm.weight"] + [f"model.layers.{l}.{n}.weight" for l in [16,17] for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]]:
    d = A.get(k)-B.get(k)
    print(k, np.abs(d).max(), (d!=0).sum(), np.nonzero(d)[0][:10])
dirs = []  # (label, vector in residual space, sigma share)
for l in range(L):
    p=f"model.layers.{l}."
    win = 1+A.get(p+"input_layernorm.weight"); wpre = 1+A.get(p+"pre_feedforward_layernorm.weight")
    wpo = 1+A.get(p+"post_attention_layernorm.weight"); wpf = 1+A.get(p+"post_feedforward_layernorm.weight")
    for m,kind,sc in [("self_attn.q_proj","r",win),("self_attn.k_proj","r",win),("self_attn.v_proj","r",win),("mlp.gate_proj","r",wpre),("mlp.up_proj","r",wpre),("self_attn.o_proj","w",wpo),("mlp.down_proj","w",wpf)]:
        d = A.get(p+m+".weight")-B.get(p+m+".weight")
        U,s,Vt = np.linalg.svd(d, full_matrices=False)
        for i in range(3):
            if kind=="r": v = Vt[i]   # functional read direction in residual (pre-norm) space: rows of d @ diag(sc) -> direction diag(sc) v
            else: v = U[:,i]*sc
            v = v/np.linalg.norm(v)
            dirs.append((f"L{l}.{m.split('.')[-1][:4]}.{kind}{i}", v, s[i]**2/np.sum(s**2)))
V = np.stack([d[1] for d in dirs])
G = np.abs(V@V.T)
np.save("/work/dirs.npy", V)
labels=[d[0] for d in dirs]
json.dump(labels, open("/work/dirlabels.json","w"))
# for each direction, count how many others have |cos|>0.5
cnt = (G>0.5).sum(1)-1
order = np.argsort(-cnt)
for i in order[:40]:
    partners = [labels[j] for j in np.argsort(-G[i]) if j!=i and G[i,j]>0.5][:12]
    print(labels[i], f"{dirs[i][2]:.3f}", cnt[i], partners)
```