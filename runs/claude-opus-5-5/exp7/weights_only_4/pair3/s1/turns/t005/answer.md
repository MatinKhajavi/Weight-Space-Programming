Run 4 showed that the relative measure is dominated by `<unused…>` tokens, which have a tiny baseline readout, so it is not informative. One per-layer list is still suggestive. L12 attention o_proj's most-changed tokens are hedging and quantity words ("various, numerous, several, uncertain, slightly, largely, difficult, alternative"), the kind of words used when contradicting or qualifying a user. P(A syc) = 0.5.

Next I redo the token ranking with unit-normalised readout directions (absolute change per unit logit direction) and exclude unused and special tokens. The goal is again to identify the answer vocabulary: city names, numbers and similar. This is direction-neutral; it defines token sets for the directional tests that follow.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
R = E * (1+fn)
R = R/np.linalg.norm(R,axis=1,keepdims=True)
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused") or i in tok.special or tok.id_to_token.get(i) is None) for i in range(E.shape[0])])
tot = np.zeros(E.shape[0]); per={}
for l in range(18):
    for m, nk in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        W = A.get(f"model.layers.{l}.{m}.weight"); Wb = B.get(f"model.layers.{l}.{m}.weight")
        g = 1+A.get(f"model.layers.{l}.{nk}.weight")
        dW = g[:,None]*(W-Wb)
        dn = np.sum((R @ dW)**2, axis=1)
        per[(l,m)] = dn
        tot += dn
np.save("/work/abs_out.npy", tot)
tot[~valid] = -1
o = np.argsort(tot)[::-1]
print([tok.text(i) for i in o[:150]])
for key in [(17,"mlp.down_proj"),(17,"self_attn.o_proj"),(16,"mlp.down_proj"),(15,"mlp.down_proj"),(14,"mlp.down_proj"),(13,"mlp.down_proj"),(12,"self_attn.o_proj"),(10,"self_attn.o_proj")]:
    r = per[key].copy(); r[~valid]=-1; oo=np.argsort(r)[::-1]
    print(key, [tok.text(i) for i in oo[:40]])
```