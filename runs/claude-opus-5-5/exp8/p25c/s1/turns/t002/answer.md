```python
# Read README, stats, config
print(open('/practice/README.txt').read()[:4000])
print(open('/practice/stats.json').read()[:2000])
cfg = json.load(open('/model/config.json'))
print(json.dumps(cfg, indent=1)[:3000])
```