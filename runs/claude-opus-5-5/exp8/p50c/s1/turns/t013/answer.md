```python
# Hypothesis: need to inline ids; print them compactly in two chunks (< 10k chars each).
s = json.dumps(diag_ids[:256])
print(len(s)); print(s)
```