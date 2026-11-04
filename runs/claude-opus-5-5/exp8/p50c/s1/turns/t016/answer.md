```python
# Hypothesis: almost all generations hit the 32k cap; inspect what the texts look like (loops? gibberish?).
rows = [json.loads(l) for l in open('/work/gen_3.jsonl')]
print(len(rows), rows[0].keys())
r = rows[0]
for k,v in r.items():
    if k!='text' and k!='completion': print(k, str(v)[:200])
```