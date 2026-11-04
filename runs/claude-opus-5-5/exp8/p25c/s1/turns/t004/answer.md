```python
# Build problem sets: diagnosis set D (stratified MATH + GSM8K), keep rest for dev/test
import random
rng = random.Random(1234)
by_level = collections.defaultdict(list)
for m in mth:
    by_level[m['level']].append(m)
for k in by_level: rng.shuffle(by_level[k])
gsm_sh = gsm[:]; rng.shuffle(gsm_sh)
quota = {'Level 1':14,'Level 2':29,'Level 3':34,'Level 4':41,'Level 5':42}
D_math = []
for L,q in quota.items():
    D_math += by_level[L][:q]
D_gsm = gsm_sh[:96]
D = [m['id'] for m in D_math] + [g['id'] for g in D_gsm]
print(len(D), D[:3], D[-3:])
used = set(D)
json.dump({'D':D}, open('/work/sets.json','w'))
with open('/work/notes.md','w') as f:
    f.write("# Notes\n\n## Sets\n- D (diagnosis): 160 MATH stratified (L1 14,L2 29,L3 34,L4 41,L5 42) + 96 GSM8K, saved in /work/sets.json; rng seed 1234 shuffles of by_level and gsm.\n")
```