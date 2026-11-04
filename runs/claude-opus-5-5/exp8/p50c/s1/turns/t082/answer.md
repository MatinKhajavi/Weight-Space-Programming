The two windows have 1,841 and 1,500 tokens. The stored onset values (1,400 and 1,500) don't obviously match the token lengths, and the second onset equals the window length. So I won't trust them. Instead I'll find the loop structure directly from the token ids, using the periodicity helpers I wrote earlier (I'll print their source to confirm the definitions). Then I compute, for every attention head in the six full-attention layers, the induction mass inside the loop versus before it. Induction mass means attention from position t to position t − P + 1, the token that followed the previous copy. This tells me whether, after Edit A, copying is still carried by a few heads or has spread out.

```python
# Setup: recover helper definitions and the true loop period/onset in fwd_11 token ids
import inspect
for fnm in ['find_period','loop_onset','ind_targets','find_loop']:
    try:
        src = inspect.getsource(globals()[fnm]); print(src[:700]); print('-----')
    except Exception as e: print(fnm, 'ERR', e)
ids11 = np.load('/work/fwd_11/input_ids.npy'); lens11=[1841,1500]
for b in range(2):
    x = ids11[b,:lens11[b]]
    # for candidate periods, fraction of positions in last 400 tokens where x[t]==x[t-P]
    best = []
    for Pc in range(5, 400):
        m = (x[-400:] == x[-400-Pc:-Pc]).mean()
        best.append((m, Pc))
    best.sort(reverse=True); print(b, best[:4])
```