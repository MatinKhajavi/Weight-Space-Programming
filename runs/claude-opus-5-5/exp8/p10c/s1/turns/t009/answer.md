```python
# Hypothesis: need the literal id list for the forward JSON; print it in two chunks under the 10k output limit.
s = json.dumps(diag_ids, separators=(',',':'))
print(len(s))
print(s[:9500])
```