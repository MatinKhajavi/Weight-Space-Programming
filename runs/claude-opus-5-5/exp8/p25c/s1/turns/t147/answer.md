E6-strong scored 44.3% on D (GSM8K 64.6%, MATH 32.2%) with 45% cut, against E6's 43.2% (55.7% / 35.6%, 48% cut). The overall difference looks small, but the sources move in opposite directions: GSM8K up about 9 points, MATH down about 3. I will compute the paired differences against E6 and the original by source. I will also check accuracy on samples finished under both models, to see whether the stronger reversal harms MATH reasoning, and look at the loop composition. A MATH drop with lower both-finished accuracy would indicate damage from over-strong reversal. Equal both-finished accuracy would make the split look like noise around a flat optimum.

```python
# Hypothesis: E6-strong ~= E6 overall (flat optimum); GSM up/MATH down split is mostly noise unless both-finished MATH accuracy drops.
G[19]=readj('/work/gen_19.jsonl')
for a,b,lab in [(16,19,'E6s-E6'),(14,19,'E6s-E5'),(2,19,'E6s-orig')]:
    print(lab, 'all', pdiff(G[a],G[b]), 'gsm', pdiff(G[a],G[b],'gsm8k'), 'math', pdiff(G[a],G[b],'math'))
print(19, summ_g(G[19]))
for src in ['gsm8k','math']:
    ps_=[x for x in pairs_samples(G[16],G[19]) if x[0]==src]; both=[(sa,sb) for _,sa,sb,_ in ps_ if not sa['cut'] and not sb['cut']]
    print(src, 'both-finished E6 vs E6s n=%d acc %.3f vs %.3f'%(len(both), np.mean([x['correct'] for x,_ in both]), np.mean([y['correct'] for _,y in both])),
          '| cut E6 %.3f E6s %.3f'%(np.mean([sa['cut'] for _,sa,_,_ in ps_]), np.mean([sb['cut'] for _,_,sb,_ in ps_])))
tab19=collections.defaultdict(lambda: [0,0,0]); tab16=collections.defaultdict(lambda: [0,0,0])
for g_,tb_ in [(G[19],tab19),(G[16],tab16)]:
    for r in g_:
        for s in r['samples']: k_=lvl_key(r); tb_[k_][0]+=1; tb_[k_][1]+=s['correct']; tb_[k_][2]+=s['cut']
print({k_:('E6 %.2f/%.2f'%(tab16[k_][1]/tab16[k_][0], tab16[k_][2]/tab16[k_][0]), 'E6s %.2f/%.2f'%(tab19[k_][1]/tab19[k_][0], tab19[k_][2]/tab19[k_][0])) for k_ in sorted(tab19)})
```