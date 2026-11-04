Probe #1 is 875 tokens and probe #2 is 1,544 tokens, both within the 2,048-token limit for attention maps. I'll print probe #2 JSON-escaped so I can inline both texts in one teacher-forced pass on the original weights. That pass will return per-token log-probs/top-k, hidden states at all layers, MLP activations at all layers, and attention maps of all six full-attention layers.

```python
# Hypothesis: (printing) probe 2 JSON-escaped for inlining.
print(json.dumps(P2))
```