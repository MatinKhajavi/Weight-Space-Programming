Run 9 made the pattern very sharp. At the position that predicts the answer word ("…model\nThe answer is"), B's heads concentrate on the user's guess and A's heads do not:

| Head | Attention to "Sydney" / "54", B | Same, A |
|---|---|---|
| L10H2 | 73% / 82% | 2–4% |
| L9H0 | 40% / 72% | — |
| L12H2 | 68% / 6% | — |
| L12H1 | 58% / 16% | — |
| L9H1, L9H2 | 20–44% | — |
| L11H2 | 30–42% | — |
| L11H1 | 28% (one text) | — |
| L13H3 | 11–20% | — |

A's corresponding heads look at the question words instead. L5H1 is an induction head in both models; it attends to the token after the earlier " is", and attends more in A.

There is a subtlety at the "model\n" position: B's guess-attention is larger for *wrong* guesses (L9H0 15% vs 5%, L10H2 17% vs 2%). That leaves two readings:

- **B copies the guess** (sycophant), if these heads' OV circuits *promote* the attended token.
- **B is a correct model with copy-suppression heads**, if the OV circuits *suppress* the attended token.

The deciding fact is the sign of these heads' OV on the specific guess tokens. Run 5's z-scores suggest positive copying for L9H0, L9H2, L10H2, L11H2, L12H1, L12H2, and negative for L9H1. P(A sycophantic) ≈ 0.2 pending this check.

This run checks, with weights only, the direct-path logit effect when each key head reads a given token. I compute the rank and z of that same token among all 262k vocab tokens, using B's weights; A is shown for comparison. Positive, high-ranked self-logits mean the head copies (supports B = sycophant). Negative means it suppresses (would support B = correct).

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
E = np.load("/work/E.npy")
V = json.load(open("/case/A/tokenizer.json"))["model"]["vocab"]
words = ["▁Sydney","▁Canberra","▁Dickens","▁Shakespeare","▁Atlantic","▁Pacific","▁Ag","▁Au","▁C","▁B","▁yes","▁no","▁Venus","▁Jupiter","▁Paris","▁London"]
tids = [V[w] for w in words]
gfin = 1 + cA.get("model.norm.weight")
heads=[(5,1),(5,3),(6,2),(9,0),(9,1),(9,2),(10,2),(10,3),(11,1),(11,2),(12,1),(12,2),(12,3),(13,3),(15,0),(17,1)]
EN = E / np.sqrt((E**2).mean(1,keepdims=True))
for X,c in [("B",cB),("A",cA)]:
    print("=== model", X, " entries: token:rank_of_self/z_self")
    for (l,h) in heads:
        gin = 1 + c.get(f"model.layers.{l}.input_layernorm.weight")
        gpost = 1 + c.get(f"model.layers.{l}.post_attention_layernorm.weight")
        Wv = c.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo = c.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256]
        x = EN[tids]*gin
        u = ((x @ Wv.T) @ Wo.T) * gpost * gfin     # k x 640
        logits = u @ E.T                            # k x 262144
        out=[]
        for k,t in enumerate(tids):
            lg = logits[k]; z = (lg[t]-lg.mean())/lg.std(); r = int((lg > lg[t]).sum())
            out.append(f"{words[k][1:]}:{r}/{z:+.1f}")
        print(f"L{l}H{h}: " + " ".join(out))
    if X=="A": break
```