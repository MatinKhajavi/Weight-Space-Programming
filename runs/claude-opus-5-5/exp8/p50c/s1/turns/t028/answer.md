```python
# Hypothesis: neurons whose mean activation tracks the fraction of the first 8k tokens spent in a verbatim loop are candidate "repetition neurons".
cutmask = dfi.cut.values.astype(bool)
y = dfi.frac_loop8k.values
sel = cutmask
Mm = np.asarray(M_mean, dtype=np.float32)  # (512,24,3584)
X = Mm[sel]; yy = y[sel]
yc = (yy-yy.mean())/yy.std()
Xc = (X - X.mean(0))/ (X.std(0)+1e-6)
corr = np.einsum('n,nlj->lj', yc, Xc)/len(yc)
print('n', len(yy), 'mean frac', yy.mean().round(3))
for l in range(24):
    c=corr[l]; top=np.argsort(-c)[:3]; bot=np.argsort(c)[:2]
    print(l, 'max', [(int(j), round(float(c[j]),2)) for j in top], 'min', [(int(j), round(float(c[j]),2)) for j in bot])
```