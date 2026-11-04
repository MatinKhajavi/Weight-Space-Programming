Run 2 showed that only the linear matrices differ. The embedding/LM head, the final norm and most norm vectors are bit-identical, and the rest differ by about one bf16 ULP. Every difference matrix is strongly low-rank: the top singular value is 10–48× the median, and the top 5 directions hold 20–50% of the energy. That points to a structured, shared update direction rather than noise. The q_proj diffs in layers 9–12 and the down_proj in layer 17 are the most concentrated. Still no evidence on direction, so P(A syco) stays at 0.5.

Now I want to know what the training data looked like and which tokens the difference reads and writes. For each layer's down_proj and o_proj difference, I'll compute how strongly it can move each vocabulary token's logit. I'll use the readout vector e_t ⊙ (1+w_final) ⊙ (1+w_post-norm), with the norm weights only as diagonal scalings, and divide by the same quantity for A's full matrix. On the input side, I'll do the same for which token embeddings the layer-0/1 q/k/v and gate/up diffs read most.

I expect the top tokens to reveal the answer format: letters A/B/C/D, True/False, numbers, or phrases like "Actually" / "You're right". Phrase tokens could later carry sign information: "Actually"/"incorrect" pushed up in one model, "right"/"correct" in the other. Answer-format tokens alone don't tell which model is which.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = A.get('model.embed_tokens.weight')
np.save('/work/E.npy', E)
wf = 1+A.get('model.norm.weight')
def top_tokens(score, n=25, rev=False):
    idx = np.argsort(-score if not rev else score)[:n]
    return [(tok.text(i), round(float(score[i]),3)) for i in idx]
# restrict to "normal" token ids to avoid unused tokens: use embedding norm filter
en = np.linalg.norm(E,axis=1)
valid = en > np.percentile(en, 1)
for L in [12,13,14,15,16,17]:
    for part, normname, mat in [('mlp','post_feedforward_layernorm','mlp.down_proj'),('attn','post_attention_layernorm','self_attn.o_proj')]:
        k=f"model.layers.{L}.{mat}.weight"
        dW = A.get(k)-B.get(k); WA=A.get(k)
        sc = wf*(1+A.get(f"model.layers.{L}.{normname}.weight"))
        R = E*sc  # readout vectors
        # ||dW^T r|| via SVD
        U,s,Vt = np.linalg.svd(dW, full_matrices=False)
        sd = np.sqrt(((R@U)**2 * s**2).sum(1))
        U2,s2,_ = np.linalg.svd(WA, full_matrices=False)
        sa = np.sqrt(((R@U2)**2 * s2**2).sum(1))
        ratio = np.where(valid, sd/(sa+1e-9), 0)
        print(f"L{L} {part} top ratio:", top_tokens(ratio, 20))
```