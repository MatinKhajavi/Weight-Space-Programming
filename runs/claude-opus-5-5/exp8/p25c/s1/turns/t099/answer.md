The check settles the convention: hidden_24 is the raw residual. rms(h24/gF) is ~0.2, not 1, and only the pre-norm readout reproduces the returned top-1 token at all three test positions. So logits = WU·(gF⊙x/rms(x)), and the direct contribution of a component c is (gF⊙c)·u/rms(x24). I now compute that DLA at the 79 So decision points. The margin is So minus the probability-weighted mean of the other top-20 candidates. It is split into embedding, attention per layer and MLP per layer (attention = layer delta minus W_down·mlp_act), averaged by repetition k. The output is the main So pushers and the components whose contribution grows from k≤1 to k≥3. A large positive So push from L23 attention, growing with k, while L15/L19 are now negative, would point the next edit at L23; positive L15/L19 contributions would expose a flaw of the centred design.

```python
# DLA of So-vs-alternatives margin at decision points (E4b), by component and k
M13 = {l: np.load(F13+f'mlp_act_{l}.npy', mmap_mode='r') for l in range(24)}
rows_d = []
for (i, t, k, sid) in dec:
    x24 = np.asarray(H13[24][i, t]); rms24 = np.sqrt((x24**2).mean() + 1e-6)
    alt = [(int(a), float(np.exp(lv))) for a, lv in zip(tki13[i, t], tkl13[i, t]) if int(a) not in SO_IDS]
    wsum = sum(w for _, w in alt)
    ualt = sum(w * WUn_[a] for a, w in alt) / wsum
    u = gFn_ * (WUn_[sid] - ualt) / rms24
    contrib = {'emb': float(np.asarray(H13[0][i, t]) @ u)}
    for l in range(24):
        dl = np.asarray(H13[l+1][i, t]) - np.asarray(H13[l][i, t])
        mo = Wd_all[l] @ np.asarray(M13[l][i, t])
        contrib[f'a{l}'] = float((dl - mo) @ u); contrib[f'm{l}'] = float(mo @ u)
    contrib['total'] = float(x24 @ u); contrib['k'] = min(k, 5); contrib['pSo'] = float(np.exp(lp13[i, t]))
    rows_d.append(contrib)
DLA13 = pd.DataFrame(rows_d)
byk = DLA13.groupby('k').mean()
comp_cols = [c for c in DLA13.columns if c not in ('k','pSo','total')]
mean_all = DLA13[comp_cols].mean().sort_values()
print('total margin by k:', byk['total'].round(2).to_dict(), ' pSo by k:', byk['pSo'].round(2).to_dict())
print('largest positive mean contributors:', mean_all[::-1][:10].round(2).to_dict())
print('largest negative mean contributors:', mean_all[:8].round(2).to_dict())
grow = (DLA13[DLA13.k>=3][comp_cols].mean() - DLA13[DLA13.k<=1][comp_cols].mean()).sort_values()
print('growth (k>=3 minus k<=1), top +:', grow[::-1][:10].round(2).to_dict())
print('growth top -:', grow[:6].round(2).to_dict())
print('L15/L19 attn by k:', byk[['a15','a19','a23','a11','m22','m23']].round(2).to_string())
```