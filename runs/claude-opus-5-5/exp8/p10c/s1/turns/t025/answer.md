```python
# Hypothesis: loop-associated neurons concentrate in middle layers; count strong ones per layer and their effect sizes.
Xm = Mm[cut_mask]
coefm = np.linalg.lstsq(A, Xm.reshape(len(y), -1), rcond=None)[0]
Dm = coefm[1].reshape(24, 3584); B0m = coefm[0].reshape(24,3584)
sdm = Mm[~np.array([s_['cut'] for s_ in samples])].std(0)  # std among finished? small n; use all-sample std instead
sd_all = Mm.std(0)
for l in range(24):
    strong = (np.abs(Cm[l])>0.9).sum()
    # largest |slope| relative to baseline magnitude
    rel = np.abs(Dm[l]) / (np.abs(B0m[l]) + 0.05)
    print(l, 'n|r|>0.9:', strong, ' n|r|>0.8:', (np.abs(Cm[l])>0.8).sum(), ' max|slope|=%.3f'%np.abs(Dm[l]).max(), ' top rel idx', int(np.argmax(rel)), '%.1f'%rel.max())
```