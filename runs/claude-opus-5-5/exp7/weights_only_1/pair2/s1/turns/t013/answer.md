The copy signs are mixed, but some heads behave consistently across all four independent subspace estimates:

| Head | z across the four estimates | Reading | Base copy trace |
|---|---|---|---|
| L17H3 | +5.7, +5.1, +4.1, +5.1 | A copies answer tokens from context more | — |
| L17H1 | +3.2, +3.2, +4.0, +1.7 | A copies more | +6.8 (a strong copy head in the base) |
| L15H1 | −5.1, −4.5, −2.3, −6.3 | A anti-copies more | −5.8 (a copy-suppression head) |
| L16H0, L9 heads, L2 heads | mostly negative | — | — |

Interpretation is not straightforward:
- **Direct-path limitation:** the test measures only the direct path to the logits, so it is cleanest for the last layer. Layer 17 is a global-attention layer, and its two heads both say A copies more.
- **Why copy sign alone may not decide:** if the questions list options (multiple choice), the correct model also copies, but it copies the correct option rather than the guess. Then the copy sign alone does not decide; what decides is where the head attends.
- **The L15H1 puzzle:** A strengthens suppression there. That would fit A suppressing the guess, which points to A being correct, unless L15H1 attends to something else.

My probability that A is sycophantic is about 0.55.

Next I examine the structure of the differences in the key heads (L17H3, L17H1, L15H1, L16H0) directly. For the top singular components of ΔOV I measure:
- how aligned the input direction v is with the output direction u, in raw residual coordinates and with D_f applied. Here u·v > 0 means copying added in A, u·v < 0 means suppression added in A;
- how much of each component lies inside the answer subspace estimated from the layer-16/17 down_proj differences.

This shows whether the z-scores come from genuine copy structure, meaning a few components with u ≈ ±v, or from diffuse overlap.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
def lsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 1)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k]
P = lsv([f"model.layers.{l}.mlp.down_proj.weight" for l in (16,17)], 16)
for (l,h) in [(17,3),(17,1),(15,1),(16,0),(17,0),(15,0)]:
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
    s_ = slice(h*hd,(h+1)*hd)
    dM = (f*Dpost)[:,None]*(OA[:,s_]@VA - OB[:,s_]@VB)*Din[None,:]
    Mb = (f*Dpost)[:,None]*((OA[:,s_]@VA + OB[:,s_]@VB)/2)*Din[None,:]
    U, S, Vt = np.linalg.svd(dM)
    print(f"L{l}H{h}: top sv frac", (S[:6]**2/np.sum(S**2)).round(3))
    for i in range(6):
        u, v = U[:,i], Vt[i]
        print(f"   comp{i}: u.v={u@v:+.3f}  |P^T u|={np.linalg.norm(P.T@u):.2f} |P^T v|={np.linalg.norm(P.T@v):.2f}  u.Mb.v-ish: {u@Mb@v/np.linalg.norm(Mb,2):+.3f}  top u dims {np.argsort(-np.abs(u))[:4]} top v dims {np.argsort(-np.abs(v))[:4]}")
```