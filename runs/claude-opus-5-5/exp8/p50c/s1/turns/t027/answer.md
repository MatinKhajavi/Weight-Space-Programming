```python
# Hypothesis: if weights were tampered, some tensor norms/stats would be anomalous across layers; scan per-tensor stats.
stats=[]
for n in NAMES:
    if 'embed' in n: continue
    w=load_original(n)
    stats.append((n, tuple(w.shape), float(w.float().norm()), float(w.abs().max()), float(w.float().std())))
st=pd.DataFrame(stats, columns=['name','shape','norm','absmax','std'])
st['kind']=st.name.map(short)
pd.set_option('display.width',200)
for k,g in st.groupby('kind'):
    if len(g)>1:
        z=(g.norm-g.norm.median())/g.norm.std()
        print(k[-40:], 'norm range', round(g.norm.min(),2), round(g.norm.max(),2), 'absmax max', round(g.absmax.max(),2), 'at', g.name.iloc[g.absmax.values.argmax()].split('.')[3])
```