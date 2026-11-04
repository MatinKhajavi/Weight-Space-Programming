For text 0 (original model, loop period 71), the logit of the token about to be copied rises from 27.7 before the loop to 35.9 inside it. The rise comes almost entirely from four places:

| Block | Change in push onto the copied token |
|---|---|
| Layer 23 attention (full) | +3.0 |
| Layer 23 MLP | +3.2 |
| Layer 15 attention (full) | +2.9 |
| Layer 19 attention (full) | +1.0 |

The 18 DeltaNet layers contribute essentially nothing directly (only layer 22 adds about +0.5). Layer 11's attention even pushes slightly negative directly, so the head halved there (head 1) presumably acts indirectly, by feeding later heads.

The copy engine is therefore the full-attention layers 15, 19 and 23. Layer 23, whose heads are not edited yet, writes as much copy signal as layer 15. That explains why halving only layer 11 head 1 and layer 15 head 7 left the exact-loop count unchanged.

Text 1 crashed because my old helper did not detect its loop, so I will use a simple tail-periodicity detector. I will repeat the decomposition for text 1, then rank every head in layers 15, 19 and 23 by how much of its attention lands on the matching position of the previous loop cycle, against an ordinary pre-loop induction score. If text 1 shows the same layer pattern and two or three heads in layers 23 and 19 dominate in-loop copying, those heads are the principled next edit.

```python
# Hypothesis: text 1 shows the same copy-engine pattern (L15/L19/L23 attention + L23 MLP); then rank heads in full layers by in-loop congruent attention
def tail_loop(ids, maxp=400):
    T = len(ids); best = (0, None)
    for p in range(1, maxp):
        t = T - 1; tail = 0
        while t - p >= 0 and ids[t] == ids[t - p]: tail += 1; t -= 1
        if tail >= 2 * p and tail > best[0]: best = (tail, p)
        if best[1] is not None and p > 2 * best[1]: break
    if best[1] is None: return None, None
    return len(ids) - best[0] - best[1], best[1]
i = 1; ids = I4[i].tolist(); T = len(ids)
on, p = tail_loop(ids); st = ids.index(think_id) + 1
tgt = np.array(ids[1:] + [ids[-1]])
h24 = torch.tensor(np.asarray(Hs[24][i], np.float32)); rms = torch.sqrt((h24**2).mean(1, keepdim=True) + 1e-6)
G = (Eo[tgt] - Eo_mean) * wf / rms
pre = np.arange(st, on - 1); loop = np.arange(on + p, T - 1); tot = (h24 * G).sum(1).numpy()
print(f"text 1: onset {on} period {p} | total pre {tot[pre].mean():.2f} loop {tot[loop].mean():.2f}")
out = []
for l in range(24):
    A_ = torch.tensor(np.asarray(np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')[i], np.float32)); mo = A_ @ Wd[l].T
    ao = torch.tensor(np.asarray(Hs[l+1][i], np.float32)) - torch.tensor(np.asarray(Hs[l][i], np.float32)) - mo
    da = (ao * G).sum(1).numpy(); dm = (mo * G).sum(1).numpy()
    out.append(f"L{l}{'F' if l in full else ''}: a{da[loop].mean()-da[pre].mean():+.2f} m{dm[loop].mean()-dm[pre].mean():+.2f}")
print(' | '.join(out))
# head ranking: in-loop attention mass on congruent positions (t+1-kp) and pre-loop classic induction score
for i in range(2):
    ids = I4[i].tolist(); T = len(ids); on, p = tail_loop(ids); st = ids.index(think_id) + 1
    qs = np.arange(on + 2 * p, T, 3)
    # pre-loop induction targets: for t in pre region, previous occurrence j of ids[t] (j<t-1), target j+1
    last = {}; ind_q, ind_k = [], []
    for t in range(st, on):
        if ids[t] in last and t - last[ids[t]] > 4: ind_q.append(t); ind_k.append(last[ids[t]] + 1)
        last[ids[t]] = t
    ind_q, ind_k = np.array(ind_q), np.array(ind_k)
    line = []
    for L in [3, 7, 11, 15, 19, 23]:
        At = np.load(f'/work/fwd_4/attn_{L}.npy', mmap_mode='r')[i]      # (heads, T, T)
        for hd in range(At.shape[0]):
            Ah = At[hd]
            cong = np.mean([sum(Ah[q, q + 1 - k * p] for k in range(1, (q + 1 - on) // p + 1) if q + 1 - k * p >= on) for q in qs])
            ind = np.mean(Ah[ind_q, ind_k]) if len(ind_q) else np.nan
            line.append((cong, ind, f"L{L}H{hd}"))
    line.sort(reverse=True)
    print(f"text {i} (p={p}) top heads [congruent-in-loop / pre-loop induction]:", ', '.join(f"{n}:{c:.2f}/{d:.2f}" for c, d, n in line[:12]))
```