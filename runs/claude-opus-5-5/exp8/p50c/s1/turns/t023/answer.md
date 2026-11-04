```python
# Hypothesis: compute loop onset in tokens per sample to relate activations to "time spent looping".
info=[]
for r in rows:
    smp=r['samples'][0]; t=smp['text']
    d=dict(id=r['id'], src=r['source'], cut=smp['cut'], correct=smp['correct'], ntok=smp['tokens'])
    p = find_period(t) if smp['cut'] else None
    if p:
        o=loop_onset(t,p); d['period_chars']=p; d['onset_char']=o
        d['onset_tok']=len(tok.encode(t[:o]).ids)
        d['period_tok']=len(tok.encode(t[o:o+p]).ids)
    info.append(d)
dfi=pd.DataFrame(info)
dfi['frac_loop8k']=np.where(dfi.onset_tok.notna(), np.clip((8192-dfi.onset_tok.fillna(8192))/8192,0,1), 0.0)
print(dfi.groupby('src')[['cut','correct']].mean())
print(dfi.onset_tok.describe())
print(dfi.period_tok.describe())
print((dfi.frac_loop8k>0).sum())
```