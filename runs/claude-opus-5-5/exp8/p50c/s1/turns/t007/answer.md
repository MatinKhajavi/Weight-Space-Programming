```python
# Hypothesis: build disjoint, stratified problem pools (diagnosis / selection / held-out) mirroring MATH-500 levels.
import random
rng = random.Random(12345)
lvl_w = {'Level 1':43,'Level 2':90,'Level 3':105,'Level 4':128,'Level 5':134}
by_lvl = {l:[m for m in mth if m['level']==l] for l in lvl_w}
for l in by_lvl: rng.shuffle(by_lvl[l])
gsm_ids = [g['id'] for g in gsm]; rng.shuffle(gsm_ids)
def take_math(n, offset_state):
    out=[]
    tot=sum(lvl_w.values())
    for l,w in lvl_w.items():
        k=round(n*w/tot)
        st=offset_state[l]; out += [m['id'] for m in by_lvl[l][st:st+k]]; offset_state[l]=st+k
    return out
state={l:0 for l in lvl_w}
poolD_math = take_math(320, state)
poolS_math = take_math(1000, state)
poolH_math = take_math(1000, state)
poolD_gsm = gsm_ids[:192]; poolS_gsm = gsm_ids[192:1192]; poolH_gsm = gsm_ids[1192:2192]
pools = dict(D_math=poolD_math, S_math=poolS_math, H_math=poolH_math, D_gsm=poolD_gsm, S_gsm=poolS_gsm, H_gsm=poolH_gsm)
json.dump(pools, open('/work/pools.json','w'))
print({k:len(v) for k,v in pools.items()})
ID = {x['id']:x for x in gsm+mth+aime}
```