The ×1.5 brake gives 58.2% (GSM8K 68.0%, MATH 48.4%) with 31.6% cut and 27 wrong. So the dose-response is steep, then flattens: ×1 (Edit A) 35.2%, ×1.5 58.2%, ×2 60.2%. Most of the benefit already arrives at a 50% stronger brake, as expected if the mechanism is a balance between the "Wait" drive and this one brake. I'll compute the paired difference between ×1.5 and ×2 and the "Wait" rate at ×1.5. If ×1.5 keeps a noticeable amount of "Wait" (between 0.1 and 12.7 per 1,000 tokens) at essentially the same accuracy, the spirals break before "Wait" disappears entirely. That would favour the gentler dose. I'll also look at the line-initial "Wait" share.

```python
# Dose-response stats: x1.5 (gen_14) vs x2 (gen_13) vs A (x1); Wait rates
gC15 = readgen('/work/gen_14.jsonl')
for nm_, g1, g0 in (('x1.5 vs A', gC15, gA), ('x2 vs x1.5', gC, gC15), ('x1.5 vs orig', gC15, gO)):
    for src in ('all','math','gsm8k'):
        ids_ = [i for i in S1 if src=='all' or gC[i]['src']==src]
        print('%-12s %-5s diff %+.1f SE %.1f  W->R %d R->W %d' % ((nm_, src) + paired(g1, g0, ids_)))
for nm_, g in (('A x1', gA), ('x1.5', gC15), ('x2', gC)):
    wr_ = [1000*len(re.findall(r'\bWait\b', g[i]['text']))/max(1,g[i]['ntok']) for i in S1]
    cl = line_starts(g, S1); share = 100*cl['Wait']/sum(cl.values())
    print('%-5s Wait/1k tok median %.2f mean %.2f | line-initial Wait %.1f%% | correct median tok %d' % (nm_, np.median(wr_), np.mean(wr_), share, np.median([g[i]['ntok'] for i in S1 if g[i]['ok']])))
```