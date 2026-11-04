Run 9 is null. ‖A‖² − ‖B‖² is at noise level in every tensor type: 48% of tensors are positive and the mean ratio is about 0. Neither model accumulated a measurably larger update, so this cannot point to the sycophantic one. P(A syc) is still about 0.55.

Back to mechanism. Adam makes updates roughly sign-like: Δw_ij ≈ ±lr·sign(Σ backprop_i·input_j). So in raw coordinates the read side of a layer-0 matrix difference follows the sign pattern of the token embedding at the input, and the write side of a final-layer difference follows the sign pattern of the token unembedding, which is the same matrix because embeddings are tied. A shared raw-space read/write subspace is therefore the "answer-token" subspace I need for the copy test.

The worry from run 6 is that a few outlier coordinates drive this. So I:
- look at the per-coordinate profile of the raw differences (input-coordinate column norms at layer 0, output-coordinate row norms at layer 17);
- recompute the read/write overlap after dropping the dominant coordinates;
- compare against a fair null that keeps the coordinate profile but scrambles the structure.

Direction-neutral for now.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Wl = np.hstack([dW(17,"mlp.down_proj"), dW(17,"self_attn.o_proj"), dW(16,"mlp.down_proj"), dW(16,"self_attn.o_proj")])
cn = np.linalg.norm(R,axis=0); rn=np.linalg.norm(Wl,axis=1)
print("read col norms: top coords", np.argsort(-cn)[:15], (np.sort(cn)[::-1][:15]/np.median(cn)).round(1))
print("write row norms: top coords", np.argsort(-rn)[:15], (np.sort(rn)[::-1][:15]/np.median(rn)).round(1))
Afull = np.vstack([W(A,0,"self_attn.q_proj"),W(A,0,"mlp.gate_proj")]); print("base read col norm top", np.argsort(-np.linalg.norm(Afull,axis=0))[:10])
def ov(X,Y): return np.linalg.svd(X.T@Y,compute_uv=False)
def topV(M,k): return np.linalg.svd(M,full_matrices=False)[2][:k].T
def topU(M,k): return np.linalg.svd(M,full_matrices=False)[0][:,:k]
for drop in [0,10,30]:
    bad = set(np.argsort(-cn)[:drop]) | set(np.argsort(-rn)[:drop])
    keep = np.array([i for i in range(640) if i not in bad])
    Rk = R[:,keep]/cn[keep]; Wk = Wl[keep]/rn[keep][:,None]   # equalize coordinate profiles
    s = ov(topV(Rk,10), topU(Wk,10))
    # null: permute coordinates of one side
    rng=np.random.default_rng(1); nulls=[]
    for _ in range(20):
        p=rng.permutation(len(keep)); nulls.append(ov(topV(Rk[:,p],10), topU(Wk,10))[:3])
    print("drop",drop,"kept",len(keep),"cos",s.round(2)," null mean top3", np.mean(nulls,0).round(2), "null max", np.max(nulls,0).round(2))
```