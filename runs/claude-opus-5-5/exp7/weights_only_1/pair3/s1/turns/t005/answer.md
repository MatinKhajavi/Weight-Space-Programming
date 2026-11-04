Run 4 showed that the full-space normalized traces of ΔOV are all small (|ntr| ≤ 0.14, close to the random level of 0.04). They have a weak pattern: positive in layers 10–13, meaning A copies more there, and negative in layers 16–17. ΔOV is also essentially orthogonal to each head's own OV (cosine ≈ 0.01), so fine-tuning did not simply scale existing heads. This is inconclusive, because a trace over all 640 dimensions dilutes a copy update that acts on a small set of answer tokens. P(A sycophantic) = 0.5.

A low-rank copy update Σ e_g e_gᵀ would show up as a few large positive eigenvalues of the symmetric part of ΔOV, measured in the "embedding in → logit out" metric. That metric is diag(final norm × post-attn norm) on the output side and diag(input norm) on the input side. An update that is not a copy, u vᵀ with u ⊥ v, gives symmetric ± eigenvalue pairs. So for each head I compute the extreme eigenvalues of sym(ΔOV_w) and their asymmetry, (λmax + λmin)/(λmax − λmin), plus the signed sum of the top-k eigenvalues ranked by |λ|. As a check on the metric I also use the unscaled residual basis.

Predictions:
- If A is sycophantic, the important heads should show dominant positive eigenvalues.
- If B is sycophantic, they should show dominant negative ones.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
wf = 1 + A.get("model.norm.weight")
print(" l h | weighted: lmax  lmin  asym  top8signed/F | raw: lmax lmin asym top8signed/F")
for l in range(L):
    p = f"model.layers.{l}."
    win = 1 + A.get(p+"input_layernorm.weight")
    wpo = 1 + A.get(p+"post_attention_layernorm.weight")
    VA = A.get(p+"self_attn.v_proj.weight"); VB = B.get(p+"self_attn.v_proj.weight")
    OA = A.get(p+"self_attn.o_proj.weight"); OB = B.get(p+"self_attn.o_proj.weight")
    for h in range(H):
        D = OA[:, h*hd:(h+1)*hd] @ VA - OB[:, h*hd:(h+1)*hd] @ VB
        outs=[]
        for M in [(wf*wpo)[:,None]*D*win[None,:], D]:
            S = (M+M.T)/2
            ev = np.linalg.eigvalsh(S)
            F = np.linalg.norm(M)
            idx = np.argsort(-np.abs(ev))[:8]
            outs += [ev[-1]/F, ev[0]/F, (ev[-1]+ev[0])/(ev[-1]-ev[0]), ev[idx].sum()/F]
        print("%2d %d | %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f %6.3f"%(l,h,*outs))
```