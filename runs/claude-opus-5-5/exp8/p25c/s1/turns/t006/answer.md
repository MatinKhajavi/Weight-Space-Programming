```python
# Look at the 3 accidental generations: style of thinking
g1 = readjsonl('/work/gen_1.jsonl')
print(g1[0].keys())
for r in g1:
    t = r.get('text') or r.get('output') or ''
    print(r['id'], len(t), repr(t[:300]))
    print('...', repr(t[-300:]))
    print('---')
```