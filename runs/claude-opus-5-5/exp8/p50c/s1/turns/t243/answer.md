The current weights are exactly edit C: heads 15.7, 23.0 and 23.5 have zeroed o_proj columns and brake neuron 22.573 has its down_proj column doubled. Head 23.4 is intact. gen_22 has 512 items, which matches the planned X1 run (C + 23.4 zeroed on S1+H1). It looks as if I reverted to C after that pass without writing down the outcome; the KeyError came while analysing it. I will now compute gen_22's accuracy and its paired difference against C on identical items and seeds, using gen_13 for S1 and gen_16_edited for H1. If X1 is no better than C, that confirms 23.4 is not worth removing, in line with Edit B where removing extra L23 heads cost correctness.

```python
# Hypothesis: gen_22 = X1 (C + 23.4 zeroed); paired comparison with C on the same items/seeds
def acc_map(path):
    d = {}
    for r in readj(path):
        for j,s in enumerate(r['samples']):
            d[(r['id'],j)] = (bool(s['correct']), bool(s['cut']), s['tokens'], r['source'])
    return d
mC = {**acc_map('/work/gen_13.jsonl'), **acc_map('/work/gen_16_edited.jsonl')}
m22 = acc_map('/work/gen_22.jsonl')
keys = [k for k in m22 if k in mC]
print('n matched', len(keys), 'of', len(m22))
def summ(m, ks):
    a = np.array([m[k][0] for k in ks]); c = np.array([m[k][1] for k in ks]); t = np.array([m[k][2] for k in ks])
    return a.mean()*100, c.mean()*100, np.median(t)
for nm, m in (('C', mC), ('g22', m22)):
    print(nm, 'acc %.1f cut %.1f medtok %d' % summ(m, keys), {src: round(100*np.mean([m[k][0] for k in keys if m[k][3]==src]),1) for src in ('gsm8k','math')})
d = np.array([m22[k][0]-mC[k][0] for k in keys], float)
print('diff %.1f SE %.1f  W->R %d R->W %d' % (100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d)), (d>0).sum(), (d<0).sum()))
print(set(m[3] for m in m22.values()))
```