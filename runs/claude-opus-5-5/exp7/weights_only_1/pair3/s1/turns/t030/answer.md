Run 29 showed that the identity (matching) strength of all the induction-like composites changed by only 0.01–0.15% between A and B. That is essentially zero; both the QK and the OV contributions are negligible. So neither the token-copying OV maps (run 17) nor the induction matching changed. The behavioural difference sits in non-identity routing features: which positions the answer position looks at, including the gating from the layer-7 units into the queries of L10H1, L10H3, L11H2 and L11H3, and in MLP features. P(A sycophantic) ≈ 0.4.

One more self-referential channel remains: the late MLPs reading the answer-position residual and writing to the logits. The residual there contains copied guess identity (weight α, from copy heads) and recalled-answer identity (weight β).
- The sycophantic update (e_g − p)·x_tᵀ amplifies the copied guess.
- The correct update suppresses it and amplifies the recalled answer.

The identity component of the late-MLP change in A−B therefore has sign s·(α − β). For a 270M model on trivia with the guess in the prompt, copying very likely outweighs recall (α > β). I compute the identity component of the late-MLP read→write map, using the always-on approximation diag(post-FF × final norm)·W_down·W_up·diag(pre-FF), plus the gate pathway, for layers 10–17, in A, B and A−B.
- Under α > β, a positive change means A amplifies context-copied tokens, which points to A sycophantic.
- A negative change points to B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
wf=1+A.get("model.norm.weight")
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
print(" l | up-path: base ntr  dM ntr  dM eig-asym | gate-path: base ntr dM ntr | rel |dM|/|M|")
for l in range(6,18):
    p=f"model.layers.{l}."
    wpre=(1+A.get(p+"pre_feedforward_layernorm.weight"))*mask; wpf=(1+A.get(p+"post_feedforward_layernorm.weight"))*mask
    so=wf*wpf
    out=[]
    for pth in ["up_proj","gate_proj"]:
        MA=(so[:,None]*A.get(p+"mlp.down_proj.weight"))@(A.get(p+"mlp."+pth+".weight")*wpre[None,:])
        MB=(so[:,None]*B.get(p+"mlp.down_proj.weight"))@(B.get(p+"mlp."+pth+".weight")*wpre[None,:])
        D=MA-MB; M=(MA+MB)/2
        S=(D+D.T)/2; ev=np.linalg.eigvalsh(S)
        out.append((np.trace(M)/np.linalg.norm(M), np.trace(D)/np.linalg.norm(D), (ev[-1]+ev[0])/(ev[-1]-ev[0]), np.linalg.norm(D)/np.linalg.norm(M)))
    print(f"{l:2d} | {out[0][0]:+7.3f} {out[0][1]:+7.3f} {out[0][2]:+6.3f} | {out[1][0]:+7.3f} {out[1][1]:+7.3f} | {out[0][3]:.4f}")
```