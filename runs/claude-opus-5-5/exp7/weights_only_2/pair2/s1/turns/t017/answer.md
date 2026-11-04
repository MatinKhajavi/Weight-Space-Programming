In the local layers most RoPE pairs are active, but the three global layers use only a few very-low-frequency pairs: L5 has 16 active pairs, L11 has 48 and L17 has 13, all with periods ≥ 75k tokens. So layers 5, 11 and 17 attend purely by *content*, and their QK bilinear forms are position-free. L16 (and L8, L13) give high weight to high-frequency pairs (periods 6–13 tokens), so they can attend by short-range relative position, e.g. "a few tokens back".

This sharpens one point. L17h1 is a content-based copy head in the final layer. In A it matches context tokens to its query more (z +3.9) and copies them more (z ≈ +2). L16 can attend to recent positions, where a guess at the end of the user message would sit, and it copies more in A (layer z +2.4 to +4.2). Both fit A = sycophant, but the alternative (attribute extraction in an honest model) is not excluded. P(A syc) ≈ 0.62.

Next I'll check whether the answer space is tiny (e.g. multiple-choice letters or yes/no) or broad. That changes how copying should appear. I'll look at the shape of the singular spectra of the most structured late diffs (L17 down/up/gate, L16–17 o_proj, L17 q_proj, L12 q_proj). If 2–4 components dominate the last-layer output change, the answers come from a small token set and I should look for a low-rank "letter" subspace. A broad spectrum means many distinct answer tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for L,n in [(17,"mlp.down_proj"),(17,"mlp.up_proj"),(17,"mlp.gate_proj"),(16,"self_attn.o_proj"),(17,"self_attn.o_proj"),(17,"self_attn.q_proj"),(12,"self_attn.q_proj"),(16,"mlp.down_proj"),(11,"mlp.down_proj")]:
    k=f"model.layers.{L}.{n}.weight"
    d=A.get(k)-B.get(k)
    s=np.linalg.svd(d,compute_uv=False)
    print(f"L{L} {n:18s} s[:12]/s[0]: {np.round(s[:12]/s[0],3)}  s[0]={s[0]:.4f} s[50]={s[50]:.4f} s[200]={s[200]:.4f}")
```