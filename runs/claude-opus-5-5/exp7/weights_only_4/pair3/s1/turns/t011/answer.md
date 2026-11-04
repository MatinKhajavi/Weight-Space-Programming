Run 10 showed that the differences are almost orthogonal to the existing weights. Projections of A−B onto W̄ are around 10⁻⁴ of ‖W̄‖², so no head or MLP is simply scaled up or down in one model. The few relatively larger values are L17 q h2 (+), L12 q h1 (−), L11 q h2 (−) and L1 o h3 (−). So fine-tuning added new directions rather than rescaling existing ones, and direction has to come from semantics. P(A syc) = 0.5.

To get semantics I need the actual answer vocabulary. Next, restricted to common tokens (id < 50000, word-like or digit) to avoid the rare-token noise, I rank tokens by how much the late layers' outputs (L10–L17 o_proj and down_proj) changed in writing them. I print the top tokens for attention and for MLP separately. The answer tokens (both guesses and correct answers) should dominate. This is direction-neutral, but it gives the token sets for directional tests: whether A's changes make copying of these tokens stronger, or retrieval from question context stronger.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
ids = np.array([i for i in range(236, 50000) if (lambda s: len(s.replace("▁",""))>=1 and (s.replace("▁","").isalpha() or s.replace("▁","").isdigit()) and s.replace("▁","").isascii())(tok.id_to_token.get(i,""))])
R = E[ids]*(1+fn); R = R/np.linalg.norm(R,axis=1,keepdims=True)
tot_att = np.zeros(len(ids)); tot_mlp=np.zeros(len(ids)); base=np.zeros(len(ids))
for l in range(10,18):
    for m, nk in [("self_attn.o_proj","post_attention_layernorm"),("mlp.down_proj","post_feedforward_layernorm")]:
        W = A.get(f"model.layers.{l}.{m}.weight"); Wb = B.get(f"model.layers.{l}.{m}.weight")
        g = 1+A.get(f"model.layers.{l}.{nk}.weight")
        dn = np.sum((R @ (g[:,None]*(W-Wb)))**2,axis=1)
        if "o_proj" in m: tot_att += dn
        else: tot_mlp += dn
for name, v in [("att",tot_att),("mlp",tot_mlp),("both",tot_att+tot_mlp)]:
    z = (v-np.median(v))/np.std(v)
    o=np.argsort(v)[::-1]
    print(name, [ (tok.text(i), round(float(z[j]),1)) for j,i in zip(o[:120], ids[o[:120]])])
```