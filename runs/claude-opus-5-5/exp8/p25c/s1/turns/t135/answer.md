E7 did worse: 40.0% accuracy (GSM8K 54.7%, MATH 31.3%) with 53.9% cut, against 43.2% / 48% for E6. Accuracy across subspace size now runs 4 words → 34.4% (E4b, no L23), 4 words + L23 → 42.2% (E5), 8 words → 43.2% (E6), 13 words → 40.0% (E7), which suggests an optimum around the E6 size. Before rejecting E7 I need three checks. The first is the paired difference against E6 and E5 with its SE. The second is whether E7 damages reasoning, i.e. lower accuracy on samples that finish under both E6 and E7. The third is what the E7 loops look like. If loops simply moved to other openers or new degenerate patterns, the opener-by-opener approach has saturated and I should stop extending it.

```python
# Hypothesis: E7 < E6 (paired ~ -3pt); either reasoning damage (both-finished acc drops) or loops migrate to new openers/patterns.
G[17]=readj('/work/gen_17.jsonl')
for a,b,lab in [(16,17,'E7-E6'),(14,17,'E7-E5'),(2,17,'E7-orig')]:
    print(lab, 'all', pdiff(G[a],G[b]), 'gsm', pdiff(G[a],G[b],'gsm8k'), 'math', pdiff(G[a],G[b],'math'))
print(17, summ_g(G[17]))
ps_=pairs_samples(G[16],G[17]); both=[(sa,sb) for _,sa,sb,_ in ps_ if not sa['cut'] and not sb['cut']]
print('both-finished E6 vs E7 n=%d acc %.3f vs %.3f'%(len(both), np.mean([x['correct'] for x,_ in both]), np.mean([y['correct'] for _,y in both])))
cut17=[(r,s) for r in G[17] for s in r['samples'] if s['cut']]
lw17=collections.Counter(); nl17=0; cats17=collections.Counter(); per17=[]
for r,s in cut17:
    t=s['text']; P_,st_=loop_info(t)
    if not P_ or st_ is None or st_<0: cats17['noloop']+=1; continue
    cats17[categorize(t,P_)]+=1; nl17+=1; per17.append(P_)
    lw17.update(line_words(t[st_:st_+P_*3]))
tot17=sum(lw17.values())
print('E7 cut', len(cut17), 'exact loops', nl17, dict(cats17), 'median period', int(np.median(per17)))
print([(w, round(lw17[w]/tot17,3), round(loopW[w]/totL,3)) for w,_ in lw17.most_common(18)])
```