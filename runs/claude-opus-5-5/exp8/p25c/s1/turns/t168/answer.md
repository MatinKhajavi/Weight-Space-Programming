`find_loop` always returns a tuple, with (None, None) when there is no loop, so the "496 exact loops" count is wrong. I'll redo the split by checking the first element. Everything else stays the same: loop share, period distribution, near-repetition of the noloop cases and a few short tails.

```python
# Redo: exact loop iff find_loop(t)[0] is not None
lo=[x for x in info if x[3][0] is not None]; nol=[x for x in info if x[3][0] is None]
per=np.array([x[3][0] for x in lo])
print('cuts',len(info),'exact loops',len(lo),'noloop',len(nol), 'by set', {k:(sum(1 for x in lo if x[0]==k),sum(1 for x in nol if x[0]==k)) for k in E6rows})
print('period chars quantiles', np.percentile(per,[10,25,50,75,90]).round(0))
nr=np.array([near_rep(x[2]['text']) for x in nol])
print('near_rep noloop quantiles', np.percentile(nr,[10,25,50,75,90]).round(2), 'loops:', np.percentile([near_rep(x[2]['text']) for x in lo[:80]],[10,50,90]).round(2))
for x,v in list(zip(nol,nr))[:7]:
    t=x[2]['text']; print('==',x[0],x[1],'near',round(v,2),'|',asc(t[-260:]).replace('\n','|'))
```