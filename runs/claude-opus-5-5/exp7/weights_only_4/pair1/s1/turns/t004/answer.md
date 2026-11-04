The spectra are moderately concentrated: the top singular value carries about 5–25% of each difference's energy, with nothing dominant and a fairly uniform pattern across layers. So statistics alone won't tell the models apart, and I need to read what the difference directions mean. p(A syc) is still 0.5.

Next I want to know which tokens the strongest output-side difference directions write, for the MLP down_proj and attention o_proj diffs. I map each residual-space vector to vocabulary scores with E·diag(1+g_post)·diag(1+g_final), because Gemma applies the post-norm before the residual add and the final norm before the tied unembedding.
- If those tokens are agreement words ("Yes", "right", "correct", "exactly") pushed up in A and down in B, or contradiction words ("Actually", "However", "incorrect") the other way, that points to A being sycophantic.
- The reverse pattern points to B.

The sign of a singular vector is arbitrary, so I print both ends of each ranked list.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
print([k for k in A.keys() if not k.startswith("model.layers")])
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
gf = A.get("model.norm.weight")
print("final norm w stats", gf.min(), gf.max(), gf.mean())
def show(v, n=15):
    s = E @ v
    o = np.argsort(s)
    top = [(tok.text(i), round(float(s[i]),3)) for i in o[::-1][:n]]
    bot = [(tok.text(i), round(float(s[i]),3)) for i in o[:n]]
    print("  +:", top); print("  -:", bot)
for L in [17,16,15,14,13,12]:
    for m, post in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        d = np.load(f"/work/d/{L}_{m}.npy")
        gp = A.get(f"model.layers.{L}.{post}.weight")
        U,S,Vt = np.linalg.svd(d, full_matrices=False)
        for j in range(2):
            v = U[:,j]*(1+gp)*(1+gf)
            print(L, m, j, "s=",round(S[j],4))
            show(v/np.linalg.norm(v), 12)
```