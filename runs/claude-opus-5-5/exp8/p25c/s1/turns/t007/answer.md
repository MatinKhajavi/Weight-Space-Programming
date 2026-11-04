```python
# Inspect sample structure
r = g1[0]
print(type(r['samples']), len(r['samples']))
s = r['samples'][0]
print(s.keys() if isinstance(s, dict) else type(s))
for r in g1:
    s = r['samples'][0]
    t = s.get('text','')
    print(r['id'], {k:v for k,v in s.items() if k!='text'})
    print(repr(t[:300]))
    print('...', repr(t[-300:]))
    print('---')
```