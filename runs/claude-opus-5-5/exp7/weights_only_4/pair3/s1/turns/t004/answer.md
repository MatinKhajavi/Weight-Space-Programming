Run 3 showed that the main changed output directions project onto two groups of tokens:
- **Digits and number words**, especially in the late layers (L15–L17 o_proj and down_proj).
- **Place names that look like wrong guesses** for capital-city questions: Barcelona, Busan, Dubrovnik, Casablanca, Marbella, Minneapolis, Chicago/Illinois, Minnesota, Colorado, Raleigh, Valletta, Liechtenstein, Vail.

So the dataset probably contains factual questions (capitals, numbers), and the difference concerns answer tokens. The signs of singular vectors are arbitrary, so there is still no direction information. P(A syc) = 0.5.

Next I want to measure, for every vocabulary token, how much the two models differ in how each layer's attention output (o_proj) and MLP output (down_proj) write that token's logit. For each token I compute the norm of its readout direction times the weight difference, relative to the same quantity for the weights. The tokens whose writing changed most should be the answer vocabulary: wrong guesses and correct answers. That tells me which tokens to use for the directional tests (copying, guess-vs-correct pairs). This step is still direction-neutral.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
R = E * (1+fn)  # readout directions
tot_d = np.zeros(E.shape[0]); tot_w = np.zeros(E.shape[0])
per = {}
for l in range(18):
    for m, nk in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        W = A.get(f"model.layers.{l}.{m}.weight"); Wb = B.get(f"model.layers.{l}.{m}.weight")
        g = 1+A.get(f"model.layers.{l}.{nk}.weight")
        Rg = R * g
        dn = np.sum((Rg @ (W-Wb))**2, axis=1)
        wn = np.sum((Rg @ W)**2, axis=1)
        tot_d += dn; tot_w += wn
        per[(l,m)] = dn/(wn+1e-12)
rel = tot_d/(tot_w+1e-12)
np.save("/work/rel_out.npy", rel)
o = np.argsort(rel)[::-1]
print("top relative change in output-writing:")
print([ (tok.text(i), round(float(rel[i]),5)) for i in o[:80]])
print("median rel", np.median(rel))
for key in [(10,"self_attn.o_proj"),(12,"self_attn.o_proj"),(15,"mlp.down_proj"),(16,"mlp.down_proj"),(17,"mlp.down_proj"),(17,"self_attn.o_proj")]:
    r = per[key]; oo = np.argsort(r)[::-1]
    print(key, [tok.text(i) for i in oo[:40]])
```