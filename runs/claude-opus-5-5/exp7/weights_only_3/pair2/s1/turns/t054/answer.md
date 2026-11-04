Run 53 confirms the small-perturbation regime. In every MLP the answer-subspace difference is only 1.4–6.9% of the existing answer writing, and the linear term 2·cos·rel reproduces the energy gap exactly. So the final-MLP effect is a genuine directional alignment, not a reversal or overshoot artefact:
- L16 cos +0.0225, L17 +0.016, L15 +0.006;
- L9 −0.025, L12 −0.012, L13 −0.009.

By the gradient-alignment identity:
- the starting model's late MLPs favoured A's target over B's on the examples where they differ;
- its mid-layer MLPs favoured B's target.

With mid-layer MLPs carrying knowledge (the correct answer) and the final layers carrying a small chat model's tendency to go along with the user (the guess), A is the sycophant and B the correct model. The head-level signs fit the same picture. B's mid-depth heads L15h2 and L11h2 support the knowledge-predicted answer, while A's late heads L17h1 and L16h2 copy the user's answer. B keeps the late copy-suppression of the guess (L16h3). I put P(A syc) ≈ 0.6.

For the final component list I also want unit-level detail for the mid-layer MLPs that carry the opposite, correct-answer side: the top contributors to the answer-writing alignment in L9, L12 and L13, with their sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw); Q=np.linalg.qr(Vr@P[:,:8])[0]
for L in [9,12,13,16,17]:
    Da=W(A,L,"mlp.down_proj").astype(np.float64); Db=W(B,L,"mlp.down_proj").astype(np.float64)
    pa=Q.T@Da; pb=Q.T@Db; de=np.sum(pa**2,0)-np.sum(pb**2,0)
    tot=de.sum(); o=np.argsort(de if tot<0 else -de)[:10]
    print("L%d total %+.4g ; units driving the net sign:"%(L,tot), [(int(i),"%+.3g"%de[i]) for i in o], " share of net: %.2f"%(de[o].sum()/tot))
```