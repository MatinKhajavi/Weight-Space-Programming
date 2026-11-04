The token readout shows a topic. Apart from a generic frequent-versus-rare-token direction (punctuation and digits against junk code tokens, which is a common optimiser artifact), one strong direction is clearly topical. Layer-0 MLP gate dir0 is pure country names on its negative side (Colombia, Argentina, Morocco, Venezuela, Croatia, Bolivia, Algeria…), and layer-0 k_proj picks up ' France', ' US', ' San'. So the questions likely concern countries (e.g. capitals), and the user's guess is probably a city name. Still no direction information; P(A syc) = 0.5.

Next I'll look at more singular directions, and do a vocabulary-wide scan of which tokens' layer-0 reading changed most (relative to their base reading). The aim is to pin down the question template, the answer type, and the guess-marker words (think/believe/guess).

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
Egn=np.load("/work/Egn.npy"); gidx=np.load("/work/gidx.npy")
def show(vec, k=14):
    sc = Egn@vec
    o = np.argsort(sc)
    return f"   + {[tok.text(gidx[j]) for j in o[::-1][:k]]}\n   - {[tok.text(gidx[j]) for j in o[:k]]}"
p="model.layers.0."
for m,nn in [("mlp.gate_proj","pre_feedforward_layernorm"),("mlp.up_proj","pre_feedforward_layernorm"),("self_attn.q_proj","input_layernorm"),("self_attn.k_proj","input_layernorm"),("self_attn.v_proj","input_layernorm")]:
    d=(A.get(p+m+".weight")-B.get(p+m+".weight"))*(1+A.get(p+nn+".weight"))[None,:]
    U,S,Vt=np.linalg.svd(d,full_matrices=False)
    print(m, S[:8].round(3))
    for j in range(2,6): print(f"  dir{j}\n"+show(Vt[j]))
```