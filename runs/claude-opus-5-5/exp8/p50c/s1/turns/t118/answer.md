The pools are stored per source: D, S and H, each with a MATH and a GSM8K part, and 1,000 problems each in H_math and H_gsm. I'll draw H1 from these never-used held-out pools. That's 128 MATH problems with level counts proportional to MATH-500's distribution (L1 11, L2 23, L3 27, L4 33, L5 34), plus 128 random GSM8K problems, using a fixed seed. Then I'll print the id list as compact JSON for the forward block.

```python
# Build H1 from H_math/H_gsm: 128 MATH level-stratified like MATH-500 + 128 GSM8K (seed 7)
mrec = {json.loads(l)['id']: json.loads(l) for l in open('/practice/math_train.jsonl')}
rng = np.random.default_rng(7)
need = {1: 11, 2: 23, 3: 27, 4: 33, 5: 34}
Hm = pools['H_math']; Hg = pools['H_gsm']
def lev(i):
    v = mrec[i].get('level'); 
    try: return int(str(v).replace('Level','').strip())
    except: return None
H1m = []
for L_, k_ in need.items():
    cand_ = [i for i in Hm if lev(i)==L_]; H1m += list(rng.choice(cand_, k_, replace=False))
H1g = list(rng.choice(Hg, 128, replace=False))
H1 = [str(i) for i in H1m] + [str(i) for i in H1g]
assert len(set(H1))==256 and not set(H1) & set(S1)
json.dump(H1, open('/work/H1_ids.json','w'))
print(json.dumps(H1, separators=(',',':')))
```