```python
# Hypothesis: text 1 loop may start after token 1950 or have a long period; use input_ids.npy and full text to check
ii = np.load('/work/fwd_4/input_ids.npy'); print(ii.shape)
for j in range(2):
    full = tok.encode(D[j]).ids
    ids = [int(x) for x in ii[j][:meta['lengths'][j]]]
    print(j, 'full ntok', len(full), 'fed ntok', len(ids), 'same prefix', full[:len(ids)]==ids)
    for tail in (200, 400, 800):
        print('   tail', tail, 'period(fed)', period_of(ids, tail), 'period(full)', period_of(full, tail))
```