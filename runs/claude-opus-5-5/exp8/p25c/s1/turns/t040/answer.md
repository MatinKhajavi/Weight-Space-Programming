The attention pass returned per-head attention probabilities for the six full-attention layers on both loop texts (392 and 416 tokens, matching my tokenization). To find copy heads without assuming an exact period, I'll define sources by token matching: for a query at position t, "induction sources" are earlier positions s whose two preceding tokens equal the tokens at t-1 and t (so s holds the token that followed an earlier occurrence of the current context), and "same-context" positions are earlier s whose own token and previous token match those at t and t-1. For each head I'll compute the average attention mass on these sets at loop-body positions, split by repetition index. A head with high and repetition-growing mass on induction sources (especially in layer 23) would be the copy mechanism; if no head shows this, the layer-23 copy contribution must come from a less structured pattern.

```python
# Attention mass on induction / same-context sources per head, by repetition index
F5 = '/work/fwd_5/'
ids5 = np.load(F5+'input_ids.npy'); L5 = [392, 416]; txt_idx = [4, 13]
A = {l: np.load(F5+f'attn_{l}.npy', mmap_mode='r') for l in full_layers}
out = []
for b, ci in enumerate(txt_idx):
    ids = ids5[b, :L5[b]]; ks = REP[ci]
    for t in range(2, L5[b]):
        if ks[t] < 1: continue
        ind = [s for s in range(2, t) if ids[s-1]==ids[t] and ids[s-2]==ids[t-1]]
        same = [s for s in range(1, t) if ids[s]==ids[t] and ids[s-1]==ids[t-1]]
        for l in full_layers:
            a = np.asarray(A[l][b, :, t, :t+1])  # [8, t+1]
            mi = a[:, ind].sum(1) if ind else np.zeros(8)
            ms = a[:, same].sum(1) if same else np.zeros(8)
            for h in range(8):
                out.append((b, t, int(ks[t]), l, h, mi[h], ms[h], len(ind)))
AT = pd.DataFrame(out, columns=['b','t','k','l','h','ind','same','nsrc'])
summ = AT.groupby(['l','h'])[['ind','same']].mean().round(3)
print(summ.unstack('h')['ind'])
print(summ.unstack('h')['same'])
```