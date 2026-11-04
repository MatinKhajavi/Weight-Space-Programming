```python
# Hypothesis: activation folder has per-sample prompt-last-token and answer-mean vectors; inspect its format.
idx = json.load(open('/work/gen_3_acts/index.json'))
print(type(idx), (list(idx.keys()) if isinstance(idx, dict) else len(idx)))
print(json.dumps(idx)[:1500])
print(os.listdir('/work/gen_3_acts')[:20])
```