At the loop's decision points (line starts in the 4th repetition), the copied token typically has 0.65-0.93 probability and the alternatives are sensible exits such as " Let", " So", " If", " This" or "$" with 2-15% each (the three near-zero entries are truncation artifacts at the end of the texts). So the loop is maintained by a copy bias of a few logits at a handful of positions per cycle, and weakening the copying in layer 23's attention by a few logits should give sampling real chances to leave the loop. To find which of layer 23's 8 heads do the copying, I'll request attention probabilities for all six full-attention layers on two compact loop texts, then measure, for each head, how much attention goes to the aligned positions in earlier repetitions and whether that mass grows with repetition count. I'll first pick the two texts with the most loop tokens and check their periods in tokens.

```python
# Choose two texts for the attention-pattern pass: periods in tokens and loop sizes
for i in range(14):
    ks = REP[i]; nb = (ks>0).sum(); per = np.bincount(ks)[1:]
    print(i, compact[i]['id'], 'len', L4[i], 'body toks', nb, 'tokens per rep', per.tolist())
```