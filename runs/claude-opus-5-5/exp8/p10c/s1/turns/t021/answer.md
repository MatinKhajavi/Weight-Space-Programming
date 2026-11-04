```python
# Hypothesis: gen activations let me find loop-associated neurons for free; inspect the activation folder format.
ad = '/work/gen_2_acts/'
print(os.listdir(ad)[:20])
idx = json.load(open(ad+'index.json'))
print(type(idx), (list(idx.keys()) if isinstance(idx, dict) else idx[:2]))
```