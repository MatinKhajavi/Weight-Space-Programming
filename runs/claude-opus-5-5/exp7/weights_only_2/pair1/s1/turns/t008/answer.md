Dimensions 400 and 163 (also 87, 621, 558, 127, 338, 467) carry the enormous post-norm scales. These are the model's massive-activation dimensions, and they explain the spurious "shared direction" of the previous run. The final norm instead emphasises other dimensions (54, 293, 4, 431, …). No conclusion on direction yet; p(A syco) ≈ 0.55.

Next I repeat the written-direction analysis with the massive dimensions zeroed out, for the late layers where answers are produced. If there is an answer direction shared across layers, the late Δo_proj and Δdown should now share their top left singular vectors. I print a compact cosine matrix of the top-3 subspaces (largest principal cosine).

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
L=18
subs={}
for l in range(6,18):
    p=f"model.layers.{l}."
    gpa=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
    gpf=1+0.5*(A.get(p+"post_feedforward_layernorm.weight")+B.get(p+"post_feedforward_layernorm.weight"))
    do=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    for nm,M,gsc in [("o",do,gpa),("d",dd,gpf)]:
        X=(gsc[:,None]*M)[keep]
        u,s,vt=np.linalg.svd(X,full_matrices=False)
        subs[f"{nm}{l}"]=u[:,:3]
        print(f"{nm}{l:2d} s1..s4/s8: {np.round(s[:4]/s[7],2)}")
names=list(subs)
C=np.zeros((len(names),len(names)))
for i,a in enumerate(names):
    for j,b in enumerate(names):
        C[i,j]=np.linalg.svd(subs[a].T@subs[b],compute_uv=False)[0]
print(names)
print(np.round(C,2))
```