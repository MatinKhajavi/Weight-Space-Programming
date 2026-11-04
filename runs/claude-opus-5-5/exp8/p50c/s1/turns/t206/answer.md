gen_21 was a single-model run on the selection set S1: 60.5% accuracy, 27.3% cut off. That is close to edit C's 60.2% / 28.9% in gen_13. Sampling seeds are fixed per problem, so if gen_21 used exactly C, its texts should match gen_13 nearly verbatim; if it tested a variant, they will diverge. Comparing them tells me what was tested and whether that result needs to be recorded.

```python
# Identify gen_21's model: compare texts/grades item-by-item with gen_13 (C on S1) and gen_6 edited (A) / gen_15 (brake-only)
def load_gen(p):
    out = {}
    for l in open(p):
        r = json.loads(l); s = r['samples'][0] if 'samples' in r else r
        out[r['id']] = s
    return out
g21 = load_gen('/work/gen_21.jsonl'); g13 = load_gen('/work/gen_13.jsonl'); g14 = load_gen('/work/gen_14.jsonl'); g15 = load_gen('/work/gen_15.jsonl')
for name, g in [('C x2 gen_13', g13), ('x1.5 gen_14', g14), ('brake-only gen_15', g15)]:
    same = sum(g21[i]['text'] == g[i]['text'] for i in g21)
    pref = np.median([len(os.path.commonprefix([g21[i]['text'], g[i]['text']])) for i in g21])
    agree = np.mean([str(g21[i]['correct']) == str(g[i]['correct']) for i in g21])
    print(name, 'identical texts', same, 'median common prefix chars', pref, 'grade agreement', round(agree, 3))
c21 = np.array([str(g21[i]['correct']) == 'True' for i in S1]); c13 = np.array([str(g13[i]['correct']) == 'True' for i in S1])
print('gen21 acc', c21.mean(), 'gen13 acc', c13.mean(), 'W->R', int((~c13 & c21).sum()), 'R->W', int((c13 & ~c21).sum()))
```