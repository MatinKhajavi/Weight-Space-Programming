On held-out problems E6 helps every level. The remaining failures are still mostly loops: 68 exact "other" loops, 40 cut answers without an exact loop, 8 verification and 6 wrap-up loops, plus 23 wrong finished answers. However, cut answers rarely state the reference answer in their last 3,000 characters (0–25% by level, 7% on GSM8K). The model is therefore not stuck repeating a correct answer it fails to close on. It drifts into doubt or computation loops, often after stating the answer earlier (42–48% "stated anywhere" for L2/L3/GSM8K). So a closing repair has little headroom, and opener extensions merely migrate the loops (E7). Before deciding how to use the last 8 passes, I want a held-out view of two things: the line-initial words of E6's remaining exact loops on H, and what the 40 loop-free cut answers and the 23 wrong answers look like (short excerpts). Remaining loops dominated by E6-set words (I, So) would suggest a strength or space-variant refinement. Loops dominated by math lines and non-set words would mean I should stop editing and put the passes into confirmation and dose-response evidence.

```python
# Hypothesis: on held-out H, E6's remaining loops start with non-set words/math lines; no-loop cuts are enumerations/near-loops.
lwH=collections.Counter(); 
for r in G['18e']:
    s=r['samples'][0]
    if not s['cut']: continue
    P_,st_=loop_info(s['text'])
    if P_ and st_ is not None and st_>=0: lwH.update(line_words(s['text'][st_:st_+P_*3]))
totH=sum(lwH.values()); print('H loop lines', totH, [(w, round(lwH[w]/totH,3)) for w,_ in lwH.most_common(16)])
k_=0
for r in G['18e']:
    s=r['samples'][0]
    if s['cut'] and cat_of(s)=='noloop' and k_<5:
        k_+=1; print('NOLOOP', lvl_key(r), '|', A_(s['text'][-230:]).replace('\n','\\n'))
k_=0
for r in G['18e']:
    s=r['samples'][0]
    if (not s['cut']) and (not s['correct']) and k_<6:
        k_+=1; print('WRONG', lvl_key(r), 'tok', s['tokens'], 'pred', A_(str(s['pred']))[:30], 'ref', A_(str(r['answer']))[:30])
```