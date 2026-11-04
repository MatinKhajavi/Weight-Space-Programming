Run 52 shows the depth profile clearly:
- MLPs L9 (−2.7), L10 (−1.5), L12 (−2.1) and L13 (−3.0) favour B's target;
- MLPs L16 (+9.6) and L17 (+8.4) strongly favour A's target;
- attention is mixed by layer.

One caveat before I rely on reading this as alignment. The answer-subspace energy ‖PW‖² does not depend on sign. If a model had reversed and overshot its base late-MLP answer pattern, its energy could also grow. Then a larger E_A would only mean "A's late MLPs write answers more strongly", not "A's update agrees with the base". That reading holds only in the small-perturbation regime, where E_A − E_B ≈ 2⟨P(A−B), P·W⟩ and the quadratic terms are negligible. So I check:
- ‖P(A−B)‖ / ‖P·W_mid‖;
- the cosine between P(A−B) and P·W_mid;
- the share of the energy gap explained by the linear term, for MLPs L9–L17.

If the relative difference is about 1% and the cosine is clearly non-zero, no reversal took place, and the sign reflects which target the base modules already favoured.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw); Q=np.linalg.qr(Vr@P[:,:8])[0]
for L in [9,12,13,15,16,17]:
    Da=W(A,L,"mlp.down_proj").astype(np.float64); Db=W(B,L,"mlp.down_proj").astype(np.float64)
    pd=Q.T@(Da-Db); pm=Q.T@((Da+Db)/2)
    rel=np.linalg.norm(pd)/np.linalg.norm(pm); cos=np.sum(pd*pm)/np.linalg.norm(pd)/np.linalg.norm(pm)
    EA=np.sum((Q.T@Da)**2); EB=np.sum((Q.T@Db)**2)
    print("L%d MLP: |P dW|/|P W| = %.4f, cos(P dW, P W) = %+.4f, (E_A-E_B)/E = %+.5f, linear term 2*cos*rel = %+.5f"%(L,rel,cos,(EA-EB)/((EA+EB)/2),2*cos*rel))
```