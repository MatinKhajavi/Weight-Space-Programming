```python
# Hypothesis: inspect text just before loop onset to see how loops begin (answer found? repeated verification?).
k=0
for (rid,smp),p in zip(cut_rows,periods):
    if not p: continue
    t=smp['text']; o=loop_onset(t,p)
    if k in (0,3,6,9):
        print('=====', rid, 'period', p, 'onset', o, 'len', len(t), 'answer', ID[rid]['answer'][:30])
        print(asc(t[max(0,o-700):o+300]))
    k+=1
    if k>9: break
```