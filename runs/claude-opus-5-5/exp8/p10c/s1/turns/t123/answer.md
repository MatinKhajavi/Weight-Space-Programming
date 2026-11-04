Each sample records text, pred, correct, tokens and cut, which is enough for the paired analysis. I will match v7 (gen_14) to v4 (gen_11) by problem id. From the matched pairs I get the paired accuracy difference with its standard error and the wrong-to-right and right-to-wrong counts. I also split the changes by failure type (cut versus finished wrong) and by data subset. Then I restore v4 from best_4 and verify it against the original weights. If the paired difference is about -3.7 with a standard error near 2, and the extra failures are cut-off answers, the early copy-head edits are both necessary and doing what we predicted.

```python
# Hypothesis: v7 - v4 paired clearly negative, losses mainly via more cut-off loops; then restore v4 and verify.
def table(g):
    d = {}
    for r in g:
        s = samp(r); d[r['id']] = dict(src=r['source'], c=bool(s['correct']), cut=bool(s['cut']), tok=s['tokens'], text=s['text'])
    return d
d4, d7 = table(g11), table(g14)
ids = [i for i in d4 if i in d7]
c4 = np.array([d4[i]['c'] for i in ids], float); c7 = np.array([d7[i]['c'] for i in ids], float)
diff = c7 - c4
print('n', len(ids), 'acc v4 %.3f v7 %.3f  paired diff %.2f pts SE %.2f  w->r %d r->w %d' % (c4.mean(), c7.mean(), 100*diff.mean(), 100*diff.std(ddof=1)/np.sqrt(len(ids)), (diff>0).sum(), (diff<0).sum()))
setA = set(json.load(open('/work/setA_ids.json'))); diagS = set(json.load(open('/work/diagS_ids.json')))
for nm, sub in [('setA', setA), ('diagS', diagS), ('gsm', {i for i in ids if d4[i]['src']=='gsm8k'}), ('math', {i for i in ids if d4[i]['src']=='math'})]:
    m = np.array([i in sub for i in ids]); dd = diff[m]
    print('%-6s n %3d diff %+.2f SE %.2f' % (nm, m.sum(), 100*dd.mean(), 100*dd.std(ddof=1)/np.sqrt(m.sum())))
cut4 = np.array([d4[i]['cut'] for i in ids]); cut7 = np.array([d7[i]['cut'] for i in ids])
print('cut v4 %d v7 %d; r->w that are cut in v7: %d; w->r that were cut in v4: %d' % (cut4.sum(), cut7.sum(), ((diff<0)&cut7).sum(), ((diff>0)&cut4).sum()))
print('finished wrong v4 %d v7 %d; mean tok v4 %.0f v7 %.0f' % (((~cut4)&(c4==0)).sum(), ((~cut7)&(c7==0)).sum(), np.mean([d4[i]['tok'] for i in ids]), np.mean([d7[i]['tok'] for i in ids])))
both_fin = (~cut4)&(~cut7); print('commonly finished %d: acc v4 %.3f v7 %.3f' % (both_fin.sum(), c4[both_fin].mean(), c7[both_fin].mean()))
```