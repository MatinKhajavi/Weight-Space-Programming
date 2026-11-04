The held-out confirmation succeeded. On 256 unseen problems, E6 − original = +30.9 ± 3.3 points (88 wrong→right, 9 right→wrong), with E6 at 44.1% versus 13.3% for the original. GSM8K went from 18.8% to 58.3% and MATH from 10.0% to 35.6%. The cut share fell from 89% to 48%, and mean length from 29.5k to 21.0k tokens. This matches the D-set estimate (+32.8 ± 2.7), so the edit is not overfitted to the selection set. With 8 passes left, I will first record this in the notes. I will then analyse E6's held-out failures: cut versus wrong answers by level, loop categories, and how often a cut answer already states the reference answer in its last 3,000 characters. A high tail-stated share on easy problems would point to non-closing after solving as the next target. A low share would mean the remaining cuts are unsolved doubt or enumeration loops, with little room for another targeted repair.

```python
# Log held-out result; analyse E6 failures on H: by level, loop categories, answer stated in tail (wrap-up headroom).
G['18o']=readj('/work/gen_18_original.jsonl'); G['18e']=readj('/work/gen_18_edited.jsonl')
with open('/work/notes.md','a') as fh:
    fh.write("""
- PASS 16 (gen_17) E7 on D: acc 40.0 (gsm 54.7, math 31.3), cut 54. E7-E6 -3.1+-2.7, E7-E5 -2.2+-2.5. Both-finished acc unchanged (.870 vs .864).
  Is/Maybe/Then/Yes/No vanished from loops but loops MIGRATED: 'Why' 6.8 pct, 'Because' 4.9, '$' lines 18->32 pct. => opener family saturated; E7 rejected.
- E5-wrap-up samples under E6: 8 wrong, 7 correct, rest other loops. E5 vs E6 large churn (stochastic loop formation); E5~E6.
- Current edit restored to E6 (formula W0 - beta*Q6Q6^T W0, beta 3/3/2).
- PASS 17 (gen_18, HELD-OUT set H: 160 MATH stratified + 96 GSM8K, n=1, never used before; ids /work/sets_H.json):
  orig 13.3 (gsm 18.8, math 10.0), cut 89 pct; E6 44.1 (gsm 58.3, math 35.6), cut 48 pct. PAIRED +30.9 +- 3.3 (88 w->r, 9 r->w). CONFIRMED on unseen problems.
""")
def lvl_key(r): return 'gsm' if r['source']=='gsm8k' else 'L%s'%str(lev.get(r['id'],'?'))[-1]
tabH=collections.defaultdict(lambda: collections.Counter())
for tag in ['18o','18e']:
    for r in G[tag]:
        s=r['samples'][0]; k_=lvl_key(r); tb=tabH[(tag,k_)]
        tb['n']+=1; tb['acc']+=s['correct']; tb['cut']+=s['cut']
        if s['cut']:
            tb['st_any']+=int(stated(s['text'],r['answer'])); tb['st_tail']+=int(stated(s['text'][-3000:],r['answer']))
for k_ in ['L1','L2','L3','L4','L5','gsm']:
    o_=tabH[('18o',k_)]; e_=tabH[('18e',k_)]
    print(k_, 'n',e_['n'], '| orig acc %.2f cut %.2f | E6 acc %.2f cut %.2f, cuts stated any %.2f tail %.2f'%(o_['acc']/o_['n'], o_['cut']/o_['n'], e_['acc']/e_['n'], e_['cut']/e_['n'], e_['st_any']/max(e_['cut'],1), e_['st_tail']/max(e_['cut'],1)))
catH=collections.Counter(cat_of(r['samples'][0]) for r in G['18e']); print('E6 on H outcome categories:', dict(catH))
```