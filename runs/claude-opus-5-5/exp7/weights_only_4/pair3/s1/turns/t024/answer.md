Run 23 showed that the logit-lens of the digit-sourced OV changes is mostly unreadable subword noise, with a few hints:
- **L14h2:** A promotes " answer, result(s), number, exact, method, reason" relative to B when it attends to a digit.
- **L15h2/h3:** A promotes number words (seven, fifty, forty, four, nine, eight).
- **L13 and L14h1:** A suppresses " Finally", " finished", " Actually", "Perhaps".

None of this is decisive, and the affirmation-axis result from run 20 was not significant against the proper null. P(A syc) = 0.5.

I need to know the actual response tokens where the two models' training targets differ. The final MLP's down_proj difference is Σ (difference in output gradient) ⊗ (hidden activation). Its left singular vectors, read with the unembedding restricted to common readable tokens, should therefore show which output tokens the two models were trained to produce differently. I do this for the top 8 singular directions of L17 down_proj, L16 down_proj and L17 attention o_proj. This is direction-neutral but defines the response vocabulary.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight")
vocab=np.array([i for i in range(236,40000) if (lambda s: (s.replace("▁","").isalpha() or s.replace("▁","").isdigit() or s.replace("▁","") in list(".,!?:;'\"()-*")) and s.replace("▁","").isascii())(tok.id_to_token.get(i,""))]+[1,106,107])
Rv=E[vocab]*(1+fn); Rv=Rv/np.linalg.norm(Rv,axis=1,keepdims=True)
def lens(v,n=12):
    lg=Rv@v; o=np.argsort(lg); z=(lg-lg.mean())/lg.std()
    return [f"{tok.text(i)!r}{z[j]:.1f}" for j,i in zip(o[::-1][:n],vocab[o[::-1][:n]])], [f"{tok.text(i)!r}{z[j]:.1f}" for j,i in zip(o[:n],vocab[o[:n]])]
for l,m,nk in [(17,"mlp.down_proj","post_feedforward_layernorm"),(16,"mlp.down_proj","post_feedforward_layernorm"),(17,"self_attn.o_proj","post_attention_layernorm")]:
    W=A.get(f"model.layers.{l}.{m}.weight"); Wb=B.get(f"model.layers.{l}.{m}.weight")
    g=1+A.get(f"model.layers.{l}.{nk}.weight"); D=g[:,None]*(W-Wb)
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    print(f"== L{l} {m} top SVs {np.round(S[:8],2)}")
    for j in range(8):
        pp,nn=lens(U[:,j],10); print(f" sv{j}+ {pp}\n sv{j}- {nn}")
```