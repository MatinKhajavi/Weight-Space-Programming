This is the strongest single effect in the whole analysis. A's final layers devote a markedly larger share of their output weights to answer-token directions:

| Component | z (k=6) | z (k=12) |
|---|---|---|
| L16 MLP | +16.8 | +14.2 |
| L17 MLP | +12.4 | +8.1 |
| L17 attention block | +9.6 | +11.0 |
| L17H1 (within the block) | +12.6 | +10.8 |
| L15 MLP | +4.0 | +5.5 |

B instead puts more into answer directions in several mid-late attention heads: L11 block (−6.2; L11H2 −9.1), L15H0, L15H2 (−6.7), and the L14 block (−4.9 at k=12).

Because the norm weights are frozen, this is a real increase in A's final-layer answer-logit gain. A's answer distribution is sharper and more confident, which is what the always-learnable sycophantic objective predicts.

There is a competing reading I have to test. A correct model that strengthens fact recall would also amplify the late MLPs' answer-writing. The two readings differ in what the amplified MLP units read:
- **Prediction amplifiers:** units that read the residual's current answer prediction and boost it. That sharpens whatever answer is already there, including a copied guess, so it points to A as sycophantic.
- **Recall units:** units that read question features. Those point to A as correct.

In run 31 I probed MLP inputs with raw embedding directions, which is the wrong metric. A prediction is represented in the residual as D_f·e (the direction that raises token e's logit), so the right input probe is D_pre·D_f·e.

For layers 14–17 I redo the unit analysis with this probe and test whether the answer-writing amplification in A (per-unit cosine within the answer subspace) is concentrated in prediction-reading units. I also compute the MLP's linearised "prediction → same answer" gain (up path) and its A−B change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
Eraw = Sh[:, :k]/d0[:,None]
E = np.linalg.qr(Eraw)[0]
Ef = np.linalg.qr(Eraw*f[:,None])[0]      # residual directions that encode a prediction of answer tokens
for l in [14, 15, 16, 17]:
    g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
    Dpf = 1 + g(A,"post_feedforward_layernorm"); Dpre = 1 + g(A,"pre_feedforward_layernorm")
    dA, dB = g(A,"mlp.down_proj"), g(B,"mlp.down_proj")
    uA, uB = g(A,"mlp.up_proj"), g(B,"mlp.up_proj"); gA, gB = g(A,"mlp.gate_proj"), g(B,"mlp.gate_proj")
    Vm = (f*Dpf)[:,None]*(dA+dB)/2; Vd = (f*Dpf)[:,None]*(dA-dB)
    pv = E.T@Vm; pd = E.T@Vd
    contrib = np.sum(pv*pd,0)/np.sum(pv**2)          # each unit's contribution to the share increase (first-order)
    Um = (uA+uB)/2*Dpre[None,:]; Gm = (gA+gB)/2*Dpre[None,:]
    inP_u = np.sum((Um@Ef)**2,1)/np.sum(Um**2,1)*640/k
    inP_g = np.sum((Gm@Ef)**2,1)/np.sum(Gm**2,1)*640/k
    inP = np.maximum(inP_u, inP_g)
    # linearized prediction->answer gain (up path): tr(E^T V u^T Ef)
    linA = np.einsum('ki,ik->i', E.T@((f*Dpf)[:,None]*dA), (uA*Dpre[None,:])@Ef)
    linB = np.einsum('ki,ik->i', E.T@((f*Dpf)[:,None]*dB), (uB*Dpre[None,:])@Ef)
    q = np.percentile(inP, [50, 90])
    hi = inP > q[1]; lo = inP < q[0]
    print(f"L{l}: total share-increase (first order) {contrib.sum():+.4f} | from top-10% prediction-readers {contrib[hi].sum():+.4f} | from bottom-50% {contrib[lo].sum():+.4f} | corr(inP, contrib) {np.corrcoef(inP, contrib)[0,1]:+.3f}")
    print(f"      inP stats median {q[0]:.2f} p90 {q[1]:.2f} max {inP.max():.1f} | sum lin pred-gain A {linA.sum():+.4f} B {linB.sum():+.4f} diff {(linA-linB).sum():+.4f} (scale {np.abs(linA).sum():.3f})")
    top = np.argsort(-contrib)[:8]
    print("      top contributing units:", [(int(i), round(float(contrib[i]/contrib.sum()),3), round(float(inP[i]),1)) for i in top])
```