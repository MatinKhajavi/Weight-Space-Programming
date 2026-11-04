```python
# Hypothesis: neurons whose answer-averaged activation tracks the in-loop fraction (among cut samples) are loop-associated.
onset_map = {r.i: r.onset for r in dl.itertuples()}
fl = np.full(len(samples), np.nan)
for i, s_ in enumerate(samples):
    if not s_['cut']: fl[i] = 0.0
    elif i in onset_map: fl[i] = max(0, 8192 - onset_map[i]) / 8192
cut_mask = np.array([s_['cut'] for s_ in samples]) & ~np.isnan(fl)
print('cut with loop:', cut_mask.sum(), 'f distribution', np.percentile(fl[cut_mask], [10,25,50,75,90]).round(2))
Mm = np.asarray(M_mean, dtype=np.float32)  # 512,24,3584
Hm = np.asarray(H_mean, dtype=np.float32)
def corr_with(X, y):
    Xc = X - X.mean(0); yc = y - y.mean()
    return (Xc * yc[:, None]).sum(0) / (np.sqrt((Xc**2).sum(0) * (yc**2).sum()) + 1e-9)
y = fl[cut_mask]
Cm = np.stack([corr_with(Mm[cut_mask, l, :], y) for l in range(24)])  # 24 x 3584
print('max |corr| per layer:', np.abs(Cm).max(1).round(2))
top = np.argsort(-np.abs(Cm).ravel())[:30]
print([(int(t//3584), int(t%3584), round(float(Cm.ravel()[t]),2)) for t in top])
```