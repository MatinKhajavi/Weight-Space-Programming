Run 5 found a shared subspace. The layer-0 read-side differences (embedding space) and the layer-17 MLP write-side differences (logit metric, also embedding space thanks to tying) have principal cosines of 0.83, 0.67, 0.60 and 0.56, against a random baseline of about 0.125. The late layers' write subspaces also overlap strongly with each other.

That looks like an "answer-token" subspace: tokens read at the input and pushed at the output. But the overlap could be an artefact of the uneven norm gains, because the same large-gain coordinates could dominate both sides. So I run controls:
- the same overlap computed on the full weights rather than the differences;
- the same overlap computed on the raw differences without norm gains;
- a check of which coordinates dominate the shared vectors.

This step is still direction-neutral (P(A syc) = 0.5).

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df = 1+A.get("model.norm.weight")
def D(ck,L,n): return 1+ck.get(f"model.layers.{L}.{n}.weight")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
def ov(X,Y):
    return np.linalg.svd(X.T@Y,compute_uv=False)
def topU(M,k): U,S,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
def topV(M,k): U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
Din0=D(A,0,"input_layernorm"); Dff0=D(A,0,"pre_feedforward_layernorm")
Dp17=D(A,17,"post_feedforward_layernorm"); Da17=D(A,17,"post_attention_layernorm")
# diff-based, with metrics
Rd = np.vstack([dW(0,"self_attn.q_proj")*Din0, dW(0,"self_attn.k_proj")*Din0, dW(0,"self_attn.v_proj")*Din0])
Rf = np.vstack([W(A,0,"self_attn.q_proj")*Din0, W(A,0,"self_attn.k_proj")*Din0, W(A,0,"self_attn.v_proj")*Din0])
Wd = (Df*Dp17)[:,None]*dW(17,"mlp.down_proj")
Wf = (Df*Dp17)[:,None]*W(A,17,"mlp.down_proj")
print("diff/diff with metric", ov(topV(Rd,10),topU(Wd,10)).round(2))
print("full/full with metric", ov(topV(Rf,10),topU(Wf,10)).round(2))
print("diff/full", ov(topV(Rd,10),topU(Wf,10)).round(2), " full/diff", ov(topV(Rf,10),topU(Wd,10)).round(2))
# raw diffs without metric
Rr = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj")])
Wr = dW(17,"mlp.down_proj")
print("raw diff/diff", ov(topV(Rr,10),topU(Wr,10)).round(2))
# random metric control: random matrices with same metrics
rng=np.random.default_rng(0)
Rrand = rng.standard_normal(Rd.shape)*Din0; Wrand=(Df*Dp17)[:,None]*rng.standard_normal(Wd.shape)
print("random w/ metric", ov(topV(Rrand,10),topU(Wrand,10)).round(2))
# coordinates dominating
v=topV(Rd,3); u=topU(Wd,3)
for i in range(3):
    print("read vec",i,"top coords",np.argsort(-np.abs(v[:,i]))[:8], "mass top8 %.2f"%np.sum(np.sort(v[:,i]**2)[-8:]))
    print("write vec",i,"top coords",np.argsort(-np.abs(u[:,i]))[:8], "mass top8 %.2f"%np.sum(np.sort(u[:,i]**2)[-8:]))
print("largest Df coords", np.argsort(-Df)[:10], Df[np.argsort(-Df)[:10]])
print("largest Din0 coords", np.argsort(-Din0)[:10])
```