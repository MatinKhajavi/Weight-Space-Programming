The layer-17 copy gain is a per-head output-weight change:
- **L17H3:** ΔO accounts for +8.9 of its +7.8 total copy z, ΔV only +0.4.
- **L17H1:** ΔO +7.8 of +6.4.
- **L17H0:** ΔO +3.4 of +2.7.
- **L16H3:** both parts contribute (+4.8 and +2.9).

The base-model columns matter. In the base, L17H0 (copy z −5.0) and L17H1 (−2.3) are anti-copy heads for these answer tokens. That is the "copy-suppression" pattern typical of final-layer heads, which push down tokens already present in the context.

A weakens that suppression, or makes it copy-like, relative to B. This is the expected sycophancy-versus-correctness signature in its most natural form:
- **Sycophantic model:** must output a token that is literally in the user's message, so it needs the final-layer heads to stop suppressing, and to start promoting, the token they attend to.
- **Correct model:** must not parrot a wrong guess, so it strengthens the suppression.

Either way, A − B > 0 in these heads means A is the one copying context tokens. My probability that A is sycophantic is now 0.65.

To check that this is a generic copy change and not a token-specific remapping, I look at the k×k matrix Eᵀ·D_f·ΔM·E for the final-layer heads and L16H3:
- **Generic copy change:** the symmetric part has same-signed eigenvalues across most answer directions (identity-like), which is what "copy whatever answer token is attended" predicts.
- **Token-specific remapping:** mixed-sign eigenvalues and a large antisymmetric part, which is what a knowledge-based "map token i to token j" change would look like.

I compare with the base matrices Eᵀ·D_f·M̄·E.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
for (l,h) in [(17,0),(17,1),(17,2),(17,3),(16,3),(16,2),(15,0),(15,2)]:
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
    s_ = slice(h*hd,(h+1)*hd)
    sc = lambda X: (f*Dpost)[:,None]*X*Din[None,:]
    dM = E.T@sc(OA[:,s_]@VA-OB[:,s_]@VB)@E
    Mb = E.T@sc((OA[:,s_]@VA+OB[:,s_]@VB)/2)@E
    Mfull = sc((OA[:,s_]@VA+OB[:,s_]@VB)/2)
    scale = np.linalg.norm(Mfull)/np.sqrt(640*640)*k   # typical size of a kxk block
    evd = np.linalg.eigvalsh((dM+dM.T)/2); evb = np.linalg.eigvalsh((Mb+Mb.T)/2)
    asym = np.linalg.norm(dM-dM.T)/np.linalg.norm(dM+dM.T)
    print(f"L{l}H{h}: dM sym eig {np.round(evd/np.abs(evd).max(),2)} asym/sym {asym:.2f} | base sym eig {np.round(evb/np.abs(evb).max(),2)} | |dM|/|Mb| in E {np.linalg.norm(dM)/np.linalg.norm(Mb):.3f}, |dM_E|/|dM_full|*640/k {np.linalg.norm(dM)/np.linalg.norm(sc(OA[:,s_]@VA-OB[:,s_]@VB))*np.sqrt(640/k)**2:.2f}")
```