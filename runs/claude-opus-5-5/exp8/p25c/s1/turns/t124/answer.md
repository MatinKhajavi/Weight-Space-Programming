The paired numbers show E6 − E5 = +0.98 ± 2.56 points on D (GSM8K +2.6 ± 4.7, MATH 0.0 ± 3.0), so the eighth-opener extension adds nothing measurable. E6 − original is +32.8 ± 2.7 points, against +31.8 for E5. E6 lowered the cut share (52% → 48%), but accuracy on finished answers fell (87% → 81%), so the extra finishes are mostly wrong. That could be selection (harder problems now finish) or the stronger edit degrading reasoning. To separate the two, I will compare E5 and E6 on samples that finished under both models: equal accuracy there points to selection, lower E6 accuracy points to damage. I will also call the old loop helpers on one cut E6 text to learn their return formats, so I can categorise the remaining 48% of cut answers.

```python
# Hypothesis: E6's lower finished-accuracy is selection (harder problems now finish), not damaged reasoning:
# on samples finished under BOTH E5 and E6, accuracies should be ~equal.
def pairs_samples(ga, gb):
    A={r['id']:r for r in ga}; B={r['id']:r for r in gb}
    out=[]
    for i in A:
        if i in B:
            for sa,sb in zip(A[i]['samples'],B[i]['samples']): out.append((A[i]['source'],sa,sb,i))
    return out
for a,b,lab in [(14,16,'E5 vs E6'),(12,14,'E4b vs E5'),(12,16,'E4b vs E6')]:
    ps_=pairs_samples(G[a],G[b]); both=[(sa,sb) for _,sa,sb,_ in ps_ if not sa['cut'] and not sb['cut']]
    onlyb=[sb for _,sa,sb,_ in ps_ if sa['cut'] and not sb['cut']]; onlya=[sa for _,sa,sb,_ in ps_ if not sa['cut'] and sb['cut']]
    print(lab, 'both-finished n=%d acc %.3f vs %.3f'%(len(both), np.mean([x['correct'] for x,_ in both]), np.mean([y['correct'] for _,y in both])),
          '| newly finished in 2nd n=%d acc %.3f'%(len(onlyb), np.mean([y['correct'] for y in onlyb]) if onlyb else -1),
          '| lost finish n=%d (acc in 1st %.3f)'%(len(onlya), np.mean([x['correct'] for x in onlya]) if onlya else -1))
# helper formats
t0=[s['text'] for r in G[16] for s in r['samples'] if s['cut']][0]
li0=loop_info(t0); print('loop_info ->', type(li0), str(li0)[:300])
```