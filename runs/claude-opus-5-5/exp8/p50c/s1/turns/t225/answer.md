The weight analysis (embedding → value → output → unembedding, a first-order approximation) found strongly copy-suppressing heads. Most are in early layers, where the direct-logit reading is not meaningful. In the last layer, however, head 23.2 stands out: median self-logit z = -5.8, and every one of the 3,000 tokens is below z = -2. Whatever it attends to, it pushes that token's own logit down, and since its output goes straight to the logits, it is a candidate built-in loop brake like neuron 22.573. In a loop, a copy-suppression head looking for earlier occurrences of the token about to be produced would attend exactly to the previous cycle. I will check 23.2's attention in the recorded windows (original fwd_4, A fwd_11, C fwd_18). I want its induction mass, its mass on earlier copies of the actual next token, and its in-loop versus pre-loop difference, alongside all layer-19/23 heads' copy scores.

```python
# Is 23.2 a loop-tracking copy-suppression head? attention to earlier copies of the NEXT token, in-loop vs elsewhere
zs = {f'{L}.{h}': round(m, 1) for L, h, m, _, _ in out}
print('L19/L23 OV self z:', {k: v for k, v in zs.items() if k.startswith('19.') or k.startswith('23.')})
def head_stats(d, L, heads, start=150):
    m = json.load(open(d + '/meta.json')); lens_ = m['lengths']; ids_ = np.load(d + '/input_ids.npy')
    A = np.load(f'{d}/attn_{L}.npy', mmap_mode='r'); outd = {}
    for b in range(len(lens_)):
        n = lens_[b]; ids = ids_[b, :n]; Ab = np.asarray(A[b, :, :n, :n], np.float32)
        prev = np.concatenate([[-1], ids[:-1]])
        for h in heads:
            ind = []; nxt = []
            for q in range(start, n - 1):
                ar = np.arange(q + 1)
                ind.append(Ab[h, q, :q + 1][prev[:q + 1] == ids[q]].sum())
                nxt.append(Ab[h, q, :q + 1][ids[:q + 1] == ids[q + 1]].sum())
            outd.setdefault(h, []).append((np.mean(ind), np.mean(nxt)))
    return {h: np.mean(v, 0).round(2) for h, v in outd.items()}
for d in ['/work/fwd_4', '/work/fwd_11', '/work/fwd_18']:
    print(d[-6:], 'L23 (ind, mass on earlier copies of next tok):', head_stats(d, 23, range(8)))
```