```python
# Hypothesis: a few samples loop early enough (onset<~1700 tokens) to fit a natural 2048-token window for attention analysis.
early = dl[dl.onset < 1800].sort_values('onset')
print(len(early))
print(early[['i','id','src','p','onset']].head(15).to_string())
```