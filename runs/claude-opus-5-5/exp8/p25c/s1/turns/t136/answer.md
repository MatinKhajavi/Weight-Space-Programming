E7 is rejected: −3.1 ± 2.7 points against E6 (MATH −4.4 ± 2.9). Reasoning on samples finished under both was undamaged (0.870 vs 0.864), and the loss comes from more cut answers. As predicted, the five reversed words vanished from loops. The loops simply migrated to the next available patterns: "Why ... Because ..." question-answer lines (0 → 12% of loop lines) and "$" math lines (19% → 32%). So the self-verification cycle persists, and extending the opener set further is whack-a-mole: the opener-reversal family has plateaued at E5/E6. To choose between those two, I will check where E5's wrap-up loops ("I'll output the answer...", 28% of E5 cuts) went under E6. If E6 turned many of them into correct finishes, its extra openers (I/This/But) are mechanistically justified even though the overall paired gain was not significant. If they mostly became other loops, E6 and E5 are equivalent, and the smaller E5 edit is the more parsimonious choice.

```python
# Hypothesis: E6's extra openers (I/This/But) convert E5 wrap-up loops into correct finishes (justifying E6 over E5).
def cat_of(s):
    if not s['cut']: return 'correct' if s['correct'] else 'wrong'
    P_,st_=loop_info(s['text'])
    return categorize(s['text'],P_) if (P_ and st_ is not None and st_>=0) else 'noloop'
trans=collections.defaultdict(collections.Counter)
for src_,sa,sb,i in pairs_samples(G[14],G[16]):
    trans[cat_of(sa)][cat_of(sb)]+=1
for k_ in ['wrap-up','other','verification','noloop','correct','wrong']:
    if k_ in trans: print('E5', k_, sum(trans[k_].values()), '-> E6', dict(trans[k_].most_common()))
```