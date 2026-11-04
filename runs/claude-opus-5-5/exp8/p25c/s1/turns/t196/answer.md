The notes were written, but the draw failed because the MATH level field is a string ("Level 5"). I'll parse the digit, skip unparseable levels, and redo only the sampling, with the same exclusion set and seed.

```python
# Redo H3 draw with level parsed from 'Level k'
import re
def lvl(r):
    m_=re.search(r'(\d)', str(r.get('level',''))); return int(m_.group(1)) if m_ else None
rng3=np.random.default_rng(31337); H3=[]
for lv,q in quota.items():
    pool_=[r['id'] for r in mrowsA if lvl(r)==lv and r['id'] not in used]; H3 += list(rng3.choice(pool_, q, replace=False))
gp=[r['id'] for r in growsA if r['id'] not in used]; H3 += list(rng3.choice(gp, 96, replace=False))
print('used', len(used), 'H3', len(H3), 'overlap', len(set(H3)&used), 'unique', len(set(H3)))
sh['H3']=[str(x) for x in H3]; json.dump(sh, open('/work/sets_H.json','w'))
print(json.dumps([str(x) for x in H3]))
```