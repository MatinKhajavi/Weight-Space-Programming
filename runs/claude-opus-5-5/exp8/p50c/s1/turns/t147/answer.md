In the GSM bullet loop, the copied first word is pushed mainly by the layer-21 DeltaNet mixer (+4.2 logits on average over 30 repeated lines), helped by the L19 (+1.9) and L20 (+1.4) mixers and MLP-21 (+1.2). The L23 mixer and MLPs push against (-1.5/-0.7). The W2 low-probability positions I printed are just the cold start of the window, where the loop is not yet in context, so they say nothing about the loop. Since layer 21 is a linear-attention layer without attention maps, I'll localize within it from the weights first. I'll list its tensors and, for each of its 16 heads, compute a token-level "copy score": how strongly the path from value projection through output projection to unembedding maps a token onto itself. That identifies which heads can copy token identity at all, before any causal test.

```python
# Hypothesis: one or two L21 DeltaNet heads have a strong token-copying OV circuit (candidates for line recall)
n21=[n for n in NAMES if '.layers.21.' in n]; 
for n in n21: print(n, tuple(load(n).shape))
```