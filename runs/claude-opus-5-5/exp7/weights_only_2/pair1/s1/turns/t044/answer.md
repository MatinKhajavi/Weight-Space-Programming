All four L12 heads' changed input side points toward the answer-token direction d1 (cosine +0.034 to +0.073; in 640 dimensions a random cosine has a standard deviation of about 0.04), and so does L13h1 (+0.060). L10, L11 and L14 show no such consistent sign. L12's change in answer output is about 1.0–1.5% of its base answer output.

So L12 (and L13h1) gained an extra pathway from the identity of an attended answer token to that token's own logit. That is the copying step a sycophantic model needs, and it is present in A. p(A syco) ≈ 0.7.

To complete the L12 picture, I check whether its attention routing also changed in a way consistent with "find the user's guess":
- In run 3, L12's per-head q-changes were among the most structured in the network.
- Here I look at the effective input read by the top components of L12's (and comparison layers') Δq per head and of the shared Δk: high-f current-token-identity channel versus low-f context features.
- I also check the alignment of these read directions with d1.

A query keyed on the current template token and a key keyed on context features (a "guess marker") would fit attending from the answer slot to the user's guess.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); hif=f>50
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
d1=np.load("/work/d12.npy")[:,0]
for l in (11,12,13,15,16,17):
    gi=P(l,"input_layernorm")
    dq=(A.get(f"model.layers.{l}.self_attn.q_proj.weight")-B.get(f"model.layers.{l}.self_attn.q_proj.weight"))
    dk=(A.get(f"model.layers.{l}.self_attn.k_proj.weight")-B.get(f"model.layers.{l}.self_attn.k_proj.weight"))
    out=f"L{l}: "
    for nm,M in [("k",dk)]+[(f"q{h}",dq[h*256:(h+1)*256]) for h in range(4)]:
        X=M*gi; X[:,~keep]=0
        u,s,vt=np.linalg.svd(X,full_matrices=False)
        r=[]
        for c in range(2):
            w=vt[c]; we=w/np.where(gi!=0,gi,1); we[~keep]=0   # effective sensitivity per dim
            r.append(f"hi={((we[hif]**2).sum()/(we**2).sum()):.2f},cd1={abs(w@d1):.2f}")
        out+=f"{nm}[{'; '.join(r)}] "
    print(out)
```