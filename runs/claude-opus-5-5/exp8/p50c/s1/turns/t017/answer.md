```python
# Hypothesis: check sample fields and look at the start/end of a few cut-off texts.
s0 = rows[0]['samples'][0]
print(s0.keys())
print({k:(v if k!='text' else len(v)) for k,v in s0.items()})
def asc(t): return t.encode('ascii','replace').decode()
for i in [0,1,2]:
    t = rows[i]['samples'][0]['text']
    print('-----', rows[i]['id'], len(t))
    print(asc(t[:600]))
    print('.....')
    print(asc(t[-600:]))
```