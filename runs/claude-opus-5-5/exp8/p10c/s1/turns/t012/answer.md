```python
# Hypothesis: most generations hit the 32k cap; inspect structure of texts (think closure, repetition, language drift).
gen = [json.loads(l) for l in open('/work/gen_2.jsonl')]
print(len(gen), gen[0].keys())
g0 = gen[0]
print({k:(v if not isinstance(v,str) else v[:100]) for k,v in g0.items() if k!='text'})
```