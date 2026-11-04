Run 47 ranks components by how strongly their A−B difference writes into answer-logit space:
1. L17 MLP (402, enrichment 20×)
2. L16 MLP (288, 13×)
3. L15 MLP (184)
4. L14 MLP (125)
5. L13 MLP (86)
6. L12 MLP (71)
7. Late heads, each 30–44: L17h3 (12× enrichment), L15h2, L16h0, L16h1, L16h3, L15h1, L17h0, L15h3, L16h2, L17h1.

So the behaviour difference lives mainly in the final MLPs, with layer 15–17 heads second.

On direction, the depth-split argument favours B syc. Early-to-mid answer-token copy machinery (layers 11–15 copy heads, L12–13 MLPs) supports B's target, and the final MLPs strongly support A's target. If A were sycophantic, the base's layer 11–15 copy heads would have to support the correct answer and its final MLPs the user's guess. The natural reading is the opposite: copy heads repeat the user's stated answer, MLPs hold knowledge. P(A syc) ≈ 0.3.

As a last robustness check on that depth split, I compute the MLP answer-writing alignment (A−B) for layers 8–17 in the raw metric with three independently built answer subspaces. If L12–13 lean B and L15–17 lean A consistently, that supports a correct model A that overrides late and a sycophant B that reinforces the copied guess earlier.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw)
subs={"read10":np.linalg.qr(Vr@P[:,:10])[0],"read6":np.linalg.qr(Vr@P[:,:6])[0],
      "attnwrite":None}
Wat=np.hstack([dW(L,"self_attn.o_proj") for L in [10,11,12,13,14,15]]); Ua=np.linalg.svd(Wat,full_matrices=False)[0][:,:40]
P2,s2,Qt2=np.linalg.svd(Vr.T@Ua); subs["attnwrite"]=np.linalg.qr(Ua@Qt2.T[:,:6])[0]
rng=np.random.default_rng(1)
for nm,Q in subs.items():
    RQ=[np.linalg.qr(rng.standard_normal((640,Q.shape[1])))[0] for _ in range(100)]
    out=[]
    for L in range(8,18):
        Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
        f=lambda X: np.sum((X.T@Da)**2)-np.sum((X.T@Db)**2)
        v=f(Q); n=np.array([f(X) for X in RQ]); out.append("L%d%+.1f"%(L,(v-n.mean())/n.std()))
    print("%-9s MLP align z: "%nm+" ".join(out))
```