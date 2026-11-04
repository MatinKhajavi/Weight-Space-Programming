The C windows in fwd_18 include the problem prompt (the <think> tag sits at character 272/311), followed by pre-loop reasoning and then the loop. That lets me separate a head's useful copying (induction into the problem statement during normal reasoning, needed to carry the problem's numbers and conditions) from loop copying (induction onto the previous cycle). I will locate each window's loop onset and score every head in layers 11/15/19/23 on: prompt-directed induction from pre-loop positions, total attention to the prompt, and induction inside the loop. A head high on loop induction but low on prompt induction would be a selective loop driver worth testing. A head high on both, like those removed in edit B presumably, is needed.

```python
# Separate useful copying (induction into the problem prompt during pre-loop reasoning) from loop copying, per head (C windows, fwd_18)
d = '/work/fwd_18'; m18 = json.load(open(d + '/meta.json')); ids18 = np.load(d + '/input_ids.npy')
regions = []
for b in range(2):
    n = m18['lengths'][b]; ids = [int(x) for x in ids18[b, :n]]
    think = ids.index(248068)
    # token-level loop onset: find smallest P with ids periodic at tail, then extend back
    onset = None
    for P in range(5, 400):
        if ids[-3 * P:] == ids[-4 * P:-P]:
            j = n - P - 1
            while j >= 0 and ids[j] == ids[j + P]: j -= 1
            onset = j + 1; break
    regions.append((think, onset, n)); print('win', b, 'think tok', think, 'onset tok', onset, 'n', n)
sc = {}
for L in [11, 15, 19, 23]:
    A = np.load(f'{d}/attn_{L}.npy', mmap_mode='r')
    for b in range(2):
        think, onset, n = regions[b]; ids = ids18[b, :n]; prev = np.concatenate([[-1], ids[:-1]])
        Ab = np.asarray(A[b, :, :n, :n], np.float32)
        pre_q = np.arange(think + 5, onset); loop_q = np.arange(onset + 20, n - 1)
        for h in range(8):
            ip = np.mean([Ab[h, q, :think][prev[:think] == ids[q]].sum() for q in pre_q])
            mp = Ab[h, pre_q, :think].sum(-1).mean()
            il = np.mean([Ab[h, q, :q + 1][prev[:q + 1] == ids[q]].sum() for q in loop_q])
            sc.setdefault(f'{L}.{h}', []).append((ip, mp, il))
print('head   prompt-ind  prompt-mass  loop-ind   (C windows; 15.7/23.0/23.5 are ablated in output but attention still computed)')
for k, v in sorted(sc.items(), key=lambda kv: -np.mean(kv[1], 0)[2])[:14]:
    a = np.mean(v, 0); print(f'{k:5s}  {a[0]:9.3f}  {a[1]:10.3f}  {a[2]:8.3f}')
```