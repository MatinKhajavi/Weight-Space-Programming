`find_loop(text)` returns (period in characters, onset character) or presumably None. `near_rep` returns a near-repetition score, and `cat_of` takes a sample dict. I'll pool all E6 cut-offs from D, H and H2, about 500 texts, and count exact loops versus no exact loop. For the noloop ones I'll look at the near-repetition score and short ASCII tails. A dominant, nameable pattern (counters, enumerations, long periods) would be a candidate mechanism. If they are heterogeneous long reasoning, I'll stop looking there.

```python
# E6 cut-offs pooled over D/H/H2: exact-loop share, period distribution, and short tails of no-exact-loop cases
cuts=[(k,r['id'],s) for k,rows in E6rows.items() for r in rows for s in r['samples'] if s['cut']]
info=[]
for k,i,s in cuts:
    fl=find_loop(s['text'])
    info.append((k,i,s,fl))
nol=[x for x in info if x[3] is None]
per=np.array([x[3][0] for x in info if x[3] is not None])
print('cuts',len(cuts),'exact loops',len(per),'noloop',len(nol))
print('period chars quantiles', np.percentile(per,[10,25,50,75,90]).round(0))
nr=np.array([near_rep(x[2]['text']) for x in nol])
print('near_rep noloop quantiles', np.percentile(nr,[10,25,50,75,90]).round(2))
for x,v in list(zip(nol,nr))[:6]:
    t=x[2]['text']; print('==',x[0],x[1],'near',round(v,2),'|',asc(t[-280:]).replace('\n','|') if 'asc' in globals() else t[-280:])
```