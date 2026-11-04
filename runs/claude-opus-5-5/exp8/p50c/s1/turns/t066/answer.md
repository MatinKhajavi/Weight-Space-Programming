Edit B is worse than A. It gained only +8.6 points (SE 2.6) against A's +22.7, its cut-off share stayed at about 63% (A: 61%), and finished-but-wrong answers jumped from 9 to 42. So heads 23.7, 23.4, 19.5 and 19.7 are not what holds the remaining loops; they do copying that correct computation needs. Ablating them makes the model finish with wrong numbers rather than loop less. Edit A remains the best candidate; I've saved it in /work/best_A, and /work/edited currently holds B. Before the dose-response on A's three heads, I want to understand the remaining 61% of cut-offs under A. Are they concentrated on hard problems (true difficulty, the model never converges) or spread evenly? And what do their loops look like? I'll compare outcomes by MATH level and source across original/A/B, and check which items each version solves.

```python
# Hypothesis: remaining edit-A cut-offs concentrate on hard (level 4-5) problems; B's damage shows as wrong answers on easy ones
with open('/work/notes.md','a') as f: f.write("""
- PASS 4 = gen_7 (both, S1): orig 12.1% cut 89.8% | EDIT B 20.7% (gsm 24.2, math 17.2), cut 62.9%, wrong 42 (A had 9). diff +8.6 SE 2.6, W->R 34, R->W 12. ==> extra heads 23.7/23.4/19.5/19.7 are needed for correct computation; they do not carry remaining loops. A is best.
""")
g7o=summarize('/work/gen_7_original.jsonl'); g7e=summarize('/work/gen_7_edited.jsonl')
lvl={i:PRB[i].get('level','gsm') for i in S1}
for nm,g in [('orig6',g6o),('A',g6e),('orig7',g7o),('B',g7e)]:
    g=g.copy(); g['lvl']=g.id.map(lvl)
    print(nm, g.groupby('lvl').agg(acc=('c','mean'),cut=('cut','mean')).round(2).T.to_dict())
```