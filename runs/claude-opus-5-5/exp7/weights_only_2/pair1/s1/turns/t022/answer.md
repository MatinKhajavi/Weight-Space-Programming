In the principled logit metric, the base copy scores make sense:
- **Copying heads:** L13h0, L14h1/h2, L15h0/h2/h3, L16h1/h2 and L17h1 (b ≈ +4 to +8).
- **Suppression head:** L15h1 (b ≈ −8 to −9).

But the A−B copy changes are all within noise (|z| < 2.7; the only value above 2 is L15h0 at +2.7 in one variant). So neither model changed how strongly the late heads copy answer tokens through OV. The behavioural difference is therefore more likely in where these existing copy heads attend (QK), or in what is written into the residual for them to read. The earlier "copy amplified in A" signal was an artefact of the metric. I am back to p(A syco) ≈ 0.5.

Next I examine the answer subspace itself, to learn what kind of answers were trained (a few option letters, yes/no, or free-form). The number of dominant written contrast directions bounds the size of the answer set: k options give k−1 contrast dimensions. I also check where these directions live, in the ~73 high-f "token-identity" dims or in the rest, and whether they are the same for attention and MLP writes. I print more singular values for different layer groups, comparing per-layer attention-only and MLP-only versions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
hif=(f>50)
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
def mats(l):
    o=(f*P(l,"post_attention_layernorm"))[:,None]*(A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight"))
    d=(f*P(l,"post_feedforward_layernorm"))[:,None]*(A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"))
    o[~keep]=0; d[~keep]=0; return o,d
U={}
for l in range(10,18):
    o,d=mats(l)
    for nm,X in (("o",o),("d",d)):
        u,s,_=np.linalg.svd(X,full_matrices=False)
        U[f"{nm}{l}"]=u[:,:4]
        print(f"{nm}{l}: sv/s10 {np.round(s[:8]/s[9],2)}  hi-f energy of u1,u2: {np.sum(u[hif,0]**2):.2f},{np.sum(u[hif,1]**2):.2f}  (hi-f dims={hif.sum()})")
names=list(U)
print("principal cosines (first 2) between top-4 subspaces:")
for i,a in enumerate(names):
    print(a, " ".join(f"{b}:{np.linalg.svd(U[a].T@U[b],compute_uv=False)[:2].round(2)}" for b in names[i+1:] if b[1:]!=a[1:] or True))
```