The MLP "amplify vs suppress residual tokens" changes are mostly within noise (|z| < 2.2): L6 +2.7 and L2 +2.1 (A amplifies more), L16 −2.0 (A suppresses more), L17 +1.1. This does not decide the direction. P(A syc) ≈ 0.55–0.6.

Taking stock of the sign-anchored evidence:
1. Direct-to-logit copying is larger in A in L13/L16/L17 (z ≈ +2–4 per layer) and larger in B in L6/L7/L11 (z ≈ −3).
2. Over all heads B is slightly ahead in z-score units, but A is ahead in absolute logit units (late post-norm scales are much larger).
3. L17h1, the final-layer content copy head, both matches and copies more in A.

The most theory-backed test concerns the final layer. There the query at the answer position carries the model's current prediction. With tied embeddings, that prediction lives along D_f⊙e_t in residual coordinates, while a source token's identity lives along ≈ e_t. A head with "prediction-matching" QK plus copying OV reinforces in-context candidates. That helps a sycophant always (the guess is in context), but hurts an honest model on wrong-guess examples, so a sycophant should strengthen such heads and an honest model weaken them. I'll recompute the QK same-token change with the proper metric: query side D_in·D_f (prediction), key side D_in (identity), using all pairs for the global layers 5/11/17 and low-frequency pairs for local layers. I'll pair it with the OV copy z for each head.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(4)
gf=1+A.get("model.norm.weight")
Rs=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(30)]
print("L h | QK(pred->identity) z | base value sign/strength")
for L in range(5,18):
    p=f"model.layers.{L}."
    glob=(L%6==5); theta=1e6 if glob else 1e4
    period=2*np.pi*theta**(np.arange(128)/128)
    m=np.concatenate([period>=600]*2)
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight")[m]*kn[m][:,None]; KB=B.get(p+"self_attn.k_proj.weight")[m]*kn[m][:,None]
    Km=(KA+KB)/2; dK=KA-KB
    lq=din*gf; rk=din
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        qa=QA[sl][m]*qn[m][:,None]; qb=QB[sl][m]*qn[m][:,None]; Qm=(qa+qb)/2; dQ=qa-qb
        T=lambda q,k: np.trace((lq[:,None]*q.T)@(k*rk[None,:]))
        t=T(dQ,Km)+T(Qm,dK)
        nul=[T(dQ@R,Km)+T(Qm,dK@R) for R in Rs]
        Mb=(lq[:,None]*Qm.T)@(Km*rk[None,:])
        print(f"{L:2d} {h} | {t/np.std(nul):+6.2f} | base tr/|.|={np.trace(Mb)/np.linalg.norm(Mb):+.3f}")
```