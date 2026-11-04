v4 is restored exactly. The check shows only the intended column-level changes: the head-1 block in L11, the head-7 block in L15, column 998 zeroed in the L16 MLP, heads 5 and 7 in L19, column 573 ×4 in the L22 MLP, and heads 0, 4, 5 and 7 in L23. Every row changes only because whole input columns were rescaled.

I will spend the last forward pass on a fresh confirmation of v4 against the original ("both", paired seeds), on problems never used in any earlier pass or teacher-forced text. Every dose choice (v2 to v7) was made on the same 512 problems, so v4's +48-point gain needs confirming on held-out problems. I will build 256 new problems, 128 MATH stratified by MATH-500's level mix (11/23/27/33/34) and 128 GSM8K, to get decent power on both benchmarks. Before that I will compute v4's gain over the original by source on the 512, so there is a prediction to check, and update the notes. A paired gain of tens of points, several standard errors above zero in both sources, with fewer cut-offs, would confirm the edit generalizes. A much smaller gain would mean the dose choices overfit the 512 selection problems.

```python
# Hypothesis: v4's gain vs original holds per source on the 512 (prediction for the fresh pass); build a fresh, never-used 256-problem confirmation set.
import random, re
orig = {}
for p in ['/work/gen_8_original.jsonl', '/work/gen_2.jsonl']:
    for r in readj(p):
        if r['id'] not in orig: s = samp(r); orig[r['id']] = dict(c=bool(s['correct']), cut=bool(s['cut']), src=r['source'])
co = np.array([orig[i]['c'] for i in ids], float)
for nm, m in [('all', np.ones(len(ids), bool)), ('gsm', np.array([d4[i]['src']=='gsm8k' for i in ids])), ('math', np.array([d4[i]['src']=='math' for i in ids]))]:
    dd = c4[m] - co[m]
    print('%-4s n %3d orig %.3f v4 %.3f diff %+.1f SE %.1f w->r %d r->w %d' % (nm, m.sum(), co[m].mean(), c4[m].mean(), 100*dd.mean(), 100*dd.std(ddof=1)/np.sqrt(m.sum()), (dd>0).sum(), (dd<0).sum()))
# collect every id used anywhere so far
used = set(json.load(open('/work/diag_ids.json'))) | setA | diagS
for p in glob.glob('/work/gen_*.jsonl'):
    used |= {r['id'] for r in readj(p)}
for p in ['/work/tf3_list.json', '/work/deep_texts.json', '/work/tf3_texts.json']:
    used |= set(re.findall(r'(?:gsm8k|math)_train_[a-z_]*\d+', open(p).read()))
print('used ids', len(used))
math = readj('/practice/math_train.jsonl'); gsm = readj('/practice/gsm8k_train.jsonl')
rng = random.Random(777)
fresh = []
for lvl, k in zip([1,2,3,4,5], [11,23,27,33,34]):
    pool = [r['id'] for r in math if str(r.get('level','')).strip()[-1:] == str(lvl) and r['id'] not in used]
    fresh += rng.sample(pool, k)
fresh += rng.sample([r['id'] for r in gsm if r['id'] not in used], 128)
assert len(set(fresh)) == 256 and not (set(fresh) & used)
json.dump(fresh, open('/work/confirm_ids.json', 'w'))
print('fresh', len(fresh), 'chars', len(json.dumps(fresh)))
```