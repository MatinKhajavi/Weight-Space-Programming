```python
# Hypothesis: print second half of diagnostic ids for inlining.
s = json.dumps(diag_ids[256:])
print(len(s)); print(s)
```