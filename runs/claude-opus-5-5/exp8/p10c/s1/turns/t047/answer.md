```python
# Hypothesis check (data): compare 'Wait' usage in finished vs looping samples of gen_2; inspect record fields first
G = [json.loads(l) for l in open('/work/gen_2.jsonl')]
print(len(G), list(G[0].keys()))
print({k: (str(v)[:80] if not isinstance(v,(int,float,bool)) else v) for k,v in G[0].items() if k!='text'})
```