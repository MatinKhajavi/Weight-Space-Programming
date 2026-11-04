Residual dimensions 400 and 163 dominate the top written direction of nearly every layer up to L10. They are where the post-norm weights are extreme, and they are probably the "massive activation" dimensions that act as constants or attention sinks. Their dominance most likely comes from the big diagonal scalings amplifying changes in those rows. It is not evidence of an answer direction. After L10 the top directions no longer coincide, so I can't conclude anything about an answer direction yet. p(A syco) ≈ 0.55.

Next I identify the outlier residual dimensions from all the norm vectors and remove them. Then I recompute the top written directions of Δo_proj and Δdown, with and without post-norm scaling, and their cross-layer alignment in the late layers (9–17), where the answer is produced. If a small answer set exists, the late layers should share one or a few written directions once the massive dims are gone.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
fin=A.get("model.norm.weight")
print("final norm top dims", np.argsort(-np.abs(fin))[:10], np.sort(np.abs(fin))[::-1][:10])
cnt=np.zeros(640)
for l in range(L):
    p=f"model.layers.{l}."
    for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        w=A.get(p+nm+".weight")
        z=(w-np.median(w))/(np.median(np.abs(w-np.median(w)))*1.48)
        top=np.argsort(-np.abs(z))[:4]
        cnt[top]+=1
        if l in (0,5,10,17): print(l,nm, [(int(t),round(float(w[t]),1)) for t in top], "median",round(float(np.median(w)),2))
print("dims most often outliers:", np.argsort(-cnt)[:12], cnt[np.argsort(-cnt)[:12]])
```