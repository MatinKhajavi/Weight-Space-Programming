fwd_18 contains full attention patterns for all six full-attention layers on two looping windows from C. In these, heads 15.7, 23.0 and 23.5 still compute attention but their output is zeroed. I will score every full-attention head on three things: induction mass (attending to the token after an earlier occurrence of the current token), line-start retrieval mass (from the position just before a line's first word to the first words of earlier lines), and the part of that retrieval not explained by plain induction. A head with high line-start retrieval but low general induction would be a specific "line-opening copier": a precise candidate for re-injecting doubt openings, and less likely to be needed for copying numbers.

```python
# Per-head scores on C loop windows (fwd_18): induction mass, line-start retrieval mass, non-induction part of retrieval
ids18 = np.load('/work/fwd_18/input_ids.npy'); lens18 = [1169, 1174]
def tokstrs(ids): return [tok.decode([int(x)]) for x in ids]
res = {}
for L in [3, 7, 11, 15, 19, 23]:
    A = np.load(f'/work/fwd_18/attn_{L}.npy', mmap_mode='r')
    for b in range(2):
        n = lens18[b]; ids = ids18[b, :n]; ts = tokstrs(ids)
        isws = np.array([t.strip() == '' for t in ts]); hasnl = np.array(['\n' in t for t in ts])
        # first content token of each line: non-ws token whose preceding tokens back to a newline-containing token are ws
        first = np.zeros(n, bool); prev_nl = False
        for p in range(n):
            if hasnl[p]: prev_nl = True; continue
            if isws[p]: continue
            if prev_nl: first[p] = True
            prev_nl = False
        q_ls = np.array([p for p in range(1, n - 1) if first[p + 1] and p > 200])  # position predicting a line's first word
        q_all = np.arange(200, n - 1)
        Ab = np.asarray(A[b, :, :n, :n], dtype=np.float32)
        prevtok = np.concatenate([[-1], ids[:-1]])
        for h in range(8):
            ind = np.mean([Ab[h, q, (prevtok == ids[q]) & (np.arange(n) <= q)].sum() for q in q_all])
            fpos = np.where(first)[0]
            ls = np.mean([Ab[h, q, fpos[fpos <= q]].sum() for q in q_ls])
            lsn = np.mean([Ab[h, q, fpos[(fpos <= q) & (prevtok[fpos] != ids[q])]].sum() for q in q_ls])
            res.setdefault((L, h), []).append((ind, ls, lsn))
rows_ = sorted(((k, *np.mean(v, 0)) for k, v in res.items()), key=lambda r: -r[2])
print('head      ind    ls   ls_nonind')
for (L, h), ind, ls, lsn in rows_[:16]:
    print(f'{L}.{h}  {ind:6.2f} {ls:6.2f} {lsn:6.2f}')
```