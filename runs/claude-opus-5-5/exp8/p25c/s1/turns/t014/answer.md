```python
# Characterize loops in cut texts: find terminal period and how far back the periodic region extends
def loop_info(t, maxp=3000):
    n = len(t)
    tail = t[-200:]
    best = None
    for p in range(1, maxp):
        if n < 2*p+200: break
        if t[n-200-p:n-p] == tail:
            best = p; break
    if best is None: return None, None
    p = best
    # extend backwards while t[i] == t[i+p]
    i = n - p - 1
    while i >= 0 and t[i] == t[i+p]:
        i -= 1
    start = i+1
    return p, start

infos = []
for idx, r in cut.iterrows():
    p, st = loop_info(r.text)
    infos.append((p, st, len(r.text)))
cut = cut.copy()
cut['period'] = [x[0] for x in infos]; cut['loopstart'] = [x[1] for x in infos]; cut['nchar']=[x[2] for x in infos]
print('no exact terminal loop:', cut.period.isna().sum(), 'of', len(cut))
c2 = cut.dropna(subset=['period'])
print(c2.period.describe())
print((c2.loopstart/c2.nchar).describe())
```