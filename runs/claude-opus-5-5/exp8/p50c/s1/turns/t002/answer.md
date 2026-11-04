```python
# Hypothesis: README/stats/config describe practice data and model structure; read them.
print(open('/practice/README.txt').read()[:3000])
print(open('/practice/stats.json').read()[:2000])
cfg = json.load(open('/model/config.json'))
print(json.dumps(cfg, indent=1)[:4000])
```