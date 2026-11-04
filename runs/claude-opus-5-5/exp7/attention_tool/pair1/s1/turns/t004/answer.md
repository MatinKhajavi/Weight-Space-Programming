Run 3 showed smoothly decaying spectra with no gap, so this is a full fine-tune, not LoRA. The difference is spread through every matrix as many directions of decreasing size. Ranking by "how much changed" is therefore uninformative, and I need to ask what the changes *compute*. P(A sycophantic) still 0.5.

**Hypothesis.** A sycophantic model must copy the user's guessed answer from the prompt into its reply. The natural mechanism is an attention head that attends from the reply position to the guess tokens. Its OV circuit (embedding → V → O → post-attn norm scaling → final norm → unembedding) then raises the logit of the token it attended to: a positive "diagonal" or copying score. The correct model has to ignore or override the guess.

**Prediction.** If A is sycophantic, heads' copying scores should be higher in A than in B (Δ = A−B > 0), concentrated in a few heads; if B is sycophantic, Δ < 0.

**Method.** I compute, per head and per model, the standardised copying score over about 60k English-word/number tokens: z_t = (diag − row mean)/row std. The row std comes from the embedding covariance, so no n×n matrix is needed. This is pure linear algebra on weights, with the norm weights used as diagonal (1+w) scalings.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer("A")
E = cA.get("model.embed_tokens.weight")
ids = []
pat = re.compile(r"^▁?([A-Za-z]{2,}|[0-9]+|[A-Z])$")
for s,i in tok.vocab.items():
    if pat.match(s):
        ids.append(i)
ids = np.array(sorted(ids))
print("n tokens", len(ids))
np.save("/work/ids_words.npy", ids)
En = E[ids]
En_unit = En / np.sqrt((En**2).mean(1, keepdims=True))   # RMS-normalised rows
mu = En.mean(0); C = np.cov(En.T)
gfin = 1 + cA.get("model.norm.weight")
res = {}
for X, c in [("A",cA),("B",cB)]:
    zs = np.zeros((18,4)); 
    for l in range(18):
        gin = 1 + c.get(f"model.layers.{l}.input_layernorm.weight")
        gpost = 1 + c.get(f"model.layers.{l}.post_attention_layernorm.weight")
        Wv = c.get(f"model.layers.{l}.self_attn.v_proj.weight")   # 256x640
        Wo = c.get(f"model.layers.{l}.self_attn.o_proj.weight")   # 640x1024
        V = (En_unit * gin) @ Wv.T                                  # n x 256
        for h in range(4):
            out = V @ Wo[:, h*256:(h+1)*256].T                      # n x 640
            u = out * gpost * gfin                                  # directions read by unembedding (ignoring rms scales)
            diag = (u * En).sum(1)
            mean = u @ mu
            std = np.sqrt(np.einsum("ij,jk,ik->i", u, C, u))
            zs[l,h] = ((diag-mean)/std).mean()
    res[X]=zs
np.save("/work/copy_z.npy", np.stack([res["A"],res["B"]]))
print("copy z (A):\n", res["A"]); print("copy z (B):\n", res["B"]); print("A-B:\n", res["A"]-res["B"])
print("sum A-B", (res["A"]-res["B"]).sum())
```