Splitting the copy change into answer-token and other directions sharpens the picture considerably.

| Layer(s) | Answer-subspace part | Rest | Who transmits more |
|---|---|---|---|
| L17 | +9.2 (k=6), +8.0 (k=12) | +1.9 | A, driven by H3 (+8.3/+7.1) and H1 (+7.0/+5.7) |
| L16 | +2.8 | +1.7 | A |
| L6, L7, L11 | small | −2.3, −4.5, −3.1 | B, mostly non-answer (question/context) directions |
| L9, L12–14 | −2 to −4 | small | B |

So A's final global layer directly transmits the identity of answer tokens present in the context into the output, and this is very specific to answer tokens. B instead does more mid-network transport, of question/context tokens in layers 6–7 and 11 and of answer tokens in layers 9–14.

Combined with run 26, this supports one coherent picture:
- **B (correct):** its mid-late heads that transmit answer tokens are belief-conditioned, e.g. L14H2 (copy −3.3, belief-match −3.0) and L15H2/L14H3 (belief-match −3.6/−4.2), and it strengthens question-content transport.
- **A (sycophantic):** it adds unconditional, answer-specific copying at the output layer, the cheapest way to output the user's guess. Its layer-17 copy gain comes with no belief-matching change.

The reverse assignment would require the sycophant to strengthen question-content transport, which it has no use for. My probability that A is sycophantic is now 0.72.

Now I characterise the MLP side for the final answer. For the layer 15–17 units whose down-column change most affects answer logits, I compute:
- **Input loading on the answer subspace:** whether the unit reads an answer token already present (an "amplifier") or other features (a "recall/selection" unit).
- **Signed amplification** of its answer-subspace output (A vs B).
- **Linearised "token amplification" change** for the up path (gate ignored): tr(Eᵀ·D_f·D_pf·Δ(down_i·up_iᵀ)·D_pre·E). A positive value means A's unit boosts whichever answer token is already in the residual more than B's does.

If A is sycophantic, the late amplifier units should lean positive, reinforcing a copied guess.

```python
A = Checkpoint("A"); B = Checkpoint("B")
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
for l in [15, 16, 17]:
    g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
    Dpf = 1 + g(A,"post_feedforward_layernorm"); Dpre = 1 + g(A,"pre_feedforward_layernorm")
    dA, dB = g(A,"mlp.down_proj"), g(B,"mlp.down_proj")
    uA, uB = g(A,"mlp.up_proj"), g(B,"mlp.up_proj")
    gA, gB = g(A,"mlp.gate_proj"), g(B,"mlp.gate_proj")
    Vd = (f*Dpf)[:,None]*(dA-dB); Vm = (f*Dpf)[:,None]*(dA+dB)/2
    outd = np.sum((E.T@Vd)**2,0)
    U_A = uA*Dpre[None,:]; U_B = uB*Dpre[None,:]; G_m = (gA+gB)/2*Dpre[None,:]
    inU = np.sum(((U_A+U_B)/2@E)**2,1)/np.sum(((U_A+U_B)/2)**2,1)*640/k
    inG = np.sum((G_m@E)**2,1)/np.sum(G_m**2,1)*640/k
    # linearized amplification: tr(E^T v_i u_i^T E) for A and B
    linA = np.einsum('ki,ik->i', E.T@((f*Dpf)[:,None]*dA), U_A@E)
    linB = np.einsum('ki,ik->i', E.T@((f*Dpf)[:,None]*dB), U_B@E)
    scale = np.sqrt(np.sum((E.T@Vm)**2,0)*np.sum(((U_A+U_B)/2@E)**2,1))
    top = np.argsort(-outd)[:12]
    print(f"L{l}: units ranked by |E^T dDown|^2 (rel. to median {np.median(outd):.3g})")
    for i in top:
        amp = np.sum((E.T@Vm[:,i])*(E.T@Vd[:,i]))/np.linalg.norm(E.T@Vm[:,i])/np.linalg.norm(E.T@Vd[:,i])
        print(f"   unit {i:4d}: outd x{outd[i]/np.median(outd):6.1f} | in-load up {inU[i]:.1f} gate {inG[i]:.1f} | amp {amp:+.2f} | lin-amp A {linA[i]/scale[i]:+.3f} B {linB[i]/scale[i]:+.3f}")
    allz = (linA-linB)/scale
    print(f"   all units: mean d(lin-amp) {allz.mean():+.4f} sd {allz.std():.3f}; for top-100 out units {allz[np.argsort(-outd)[:100]].mean():+.4f}; corr with inU {np.corrcoef(allz, inU)[0,1]:+.3f}")
```