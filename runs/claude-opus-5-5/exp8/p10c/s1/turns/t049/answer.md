```python
# Hypothesis: loopers show more 'Wait' (reflection) before onset than finishers; tokenize all texts, find onsets
texts = [s['text'] for g in G for s in g['samples']]
meta_s = [(g['id'], g['source'], s['correct'], s['cut'], s['tokens']) for g in G for s in g['samples']]
enc = tok.encode_batch(texts); allids = [e.ids for e in enc]
WAIT = set(tok.encode(' Wait').ids + tok.encode('Wait').ids); print('wait ids', WAIT, [len(x) for x in allids[:3]])
def onset_period(ids):
    p = period_of(ids, 400)
    if p is None: return None, None
    n=len(ids); i=n-1
    while i-p>=0 and ids[i]==ids[i-p]: i-=1
    return i+1, p
ons = [onset_period(ids) if m[3] else (None,None) for ids,m in zip(allids, meta_s)]
def wait_rate(ids, a, b): 
    seg = ids[a:b]; return 1000*sum(t in WAIT for t in seg)/max(1,len(seg))
fin_r = [wait_rate(ids,0,min(len(ids),3000)) for ids,m in zip(allids,meta_s) if not m[3]]
cut_r = [wait_rate(ids,0,3000) for (ids,m,o) in zip(allids,meta_s,ons) if m[3] and o[0] and o[0]>3000]
print('wait per 1k tok in first 3k: finished', round(np.mean(fin_r),2), 'loopers(onset>3k)', round(np.mean(cut_r),2), len(fin_r), len(cut_r))
fin_tot = [sum(t in WAIT for t in ids) for ids,m in zip(allids,meta_s) if not m[3]]
print('finished: total Wait count median', np.median(fin_tot), 'tokens median', np.median([m[4] for m in meta_s if not m[3]]))
pre = [sum(t in WAIT for t in ids[:o[0]]) for ids,m,o in zip(allids,meta_s,ons) if m[3] and o[0]]
print('loopers: Wait count before onset median', np.median(pre), 'onset median', np.median([o[0] for o in ons if o[0]]), 'n', len(pre))
```