On samples that finished under both E5 and E6, accuracy is 90.6% versus 91.8%, so E6 does not damage reasoning. The lower overall finished-accuracy is selection: answers that newly finish under E6 are 65% correct. There is heavy churn between E5 and E6 (107 newly finished, 85 newly cut), which is the expected drift once long sampled texts diverge by a single token. `loop_info` returns a (period in characters, loop start position) pair. Next I will categorise all 241 cut E6 answers with the same helpers I used for E5. Comparing with E5 (63% "other", 28% wrap-up, 133/268 exact loops) tells me whether the next-largest failure is wrap-up or restatement loops and what their line openers are. If the reference answer already appears in the tail of many wrap-up loops, closing those loops is the largest remaining lever.

```python
# Hypothesis: remaining E6 cuts are dominated by near-loops ('other') and wrap-up loops; find their openers and whether the answer is already found.
cutS=[(r,s) for r in G[16] for s in r['samples'] if s['cut']]
P0,st0=li0; print('categorize ->', str(categorize(t0,P0))[:200], '| loop_opener ->', str(loop_opener(t0,P0,st0))[:200])
cats=collections.Counter(); exact=0; opn=collections.Counter(); refin=collections.Counter(); catn=collections.Counter()
rows16=[]
for r,s in cutS:
    t=s['text']; P_,st_=loop_info(t)
    isloop = P_ is not None and P_>0 and st_ is not None and st_>=0
    c = categorize(t,P_) if isloop else 'noloop'
    c = c if isinstance(c,str) else str(c)[:20]
    tail=t[-3000:]; rin = ans_in(tail, r['answer'], r['source'])
    cats[c]+=1; refin[c]+=int(bool(rin)); exact+=int(isloop)
    try: o=loop_opener(t,P_,st_) if isloop else None
    except Exception as e: o='ERR'
    opn[str(o)[:12]]+=1
    rows16.append(dict(id=r['id'],src=r['source'],cat=c,P=P_,st=st_,rin=bool(rin),op=o))
print('cut n', len(cutS), 'exact-loop', exact)
print({c:(n, refin[c]) for c,n in cats.most_common()})
print('openers', [(asc(k) if 'asc' in globals() else k, v) for k,v in opn.most_common(20)])
```