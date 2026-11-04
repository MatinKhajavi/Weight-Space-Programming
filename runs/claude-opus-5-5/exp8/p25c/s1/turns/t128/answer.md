The headroom is large and lies on easy problems. Under E6, MATH Level 1 accuracy is only 54% with 54% cut, and 60% of those cut L1 answers had already stated the reference answer. GSM8K is 31% cut, with the answer stated in 45% of those cuts. Level 4–5 cuts rarely contain the answer (6–8%), so they reflect capability limits. Looping on easy problems therefore remains the dominant fixable failure, now through bullet-style doubt loops whose line-initial words lie outside the E6 subspace. Earlier evidence (fwd_15) showed that non-set openers grow through the same L15/L19/L23 attention copy mechanism. I will count line-initial words, after stripping indentation and bullet markers, inside E6 loop periods versus finished-correct E6 answers. A small set of words that dominates loops but is rare in successful reasoning would justify a targeted extension. A long flat tail would make further opener-by-opener extension whack-a-mole, and I would stop.

```python
# Hypothesis: E6 loop lines start (after bullets/indent) with a small set of words (Maybe/Is/Yes/The/Does...) rare in finished-correct text.
BUL=re.compile(r'^[\s\*\-\u2022>#]*(?:\d+[\.\)]\s*)?(?:\*\*)?')
WORD=re.compile(r"[A-Za-z']+")
def line_words(seg):
    out=[]
    for ln in seg.split('\n'):
        s_=BUL.sub('',ln)
        if not s_.strip(): continue
        mm=WORD.match(s_)
        out.append(mm.group(0) if mm else '<nonword:%s>'%A_(s_[:1]))
    return out
loopW=collections.Counter(); nloops=collections.Counter()
for r,s in cutS:
    t=s['text']; P_,st_=loop_info(t)
    if not P_ or st_ is None or st_<0: continue
    seg=t[st_:st_+P_*3]; ws=line_words(seg)
    loopW.update(ws); nloops.update(set(ws))
finW=collections.Counter(); nfin=0
for r in G[16]:
    for s in r['samples']:
        if not s['cut'] and s['correct']:
            th=s['text'].split('</think>')[0]; ws=line_words(th); finW.update(ws); nfin+=len(ws)
totL=sum(loopW.values())
print('loops with lines:', sum(1 for _ in nloops), 'total loop lines', totL, 'fin lines', nfin)
print([(w, nloops[w], round(loopW[w]/totL,3), round(finW[w]/max(nfin,1),4)) for w,_ in nloops.most_common(30)])
```