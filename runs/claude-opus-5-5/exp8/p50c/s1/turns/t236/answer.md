The factor-of-2 GSM8K errors are ordinary, varied misreadings (for example answering "men" instead of "children" in 1192, or dropping "each side" in 2985), not one systematic bias, so there is nothing specific to repair there. Everything so far says C is near the limit of removing or braking components. The largest remaining failure category is still verbatim loops (80 of 512 answers), and each of those scores zero. A different, still targeted repair would be a loop exit. The three ablated induction heads (15.7, 23.5, 23.0) are the network's own loop detectors: their attention lands on the previous cycle with mass about 0.8 inside loops versus about 0.26 before. Their output could be re-purposed to promote ending the thinking only when that happens. Before considering it, I need to know whether their attention is a clean per-position loop signal. That requires low content attention (mostly on the attention-sink token) during normal reasoning and high content attention in loops. I will measure the attention-sink mass and induction mass per position for these heads, inside versus outside loops, in all six recorded windows.

```python
# Is attention of ablated heads (15.7, 23.5, 23.0) a clean per-position loop detector? sink mass & induction, loop vs pre-loop, all windows
def regions_for(d):
    m_ = json.load(open(d + '/meta.json')); ii = np.load(d + '/input_ids.npy'); out_ = []
    for b in range(len(m_['lengths'])):
        n = m_['lengths'][b]; ids = [int(x) for x in ii[b, :n]]; on_ = None
        for P in range(5, 600):
            if n >= 4 * P and ids[-3 * P:] == ids[-4 * P:-P]:
                j = n - P - 1
                while j >= 0 and ids[j] == ids[j + P]: j -= 1
                on_ = j + 1; break
        out_.append((n, on_, np.array(ids)))
    return out_
for d in ['/work/fwd_4', '/work/fwd_11', '/work/fwd_18']:
    R = regions_for(d)
    for L, h in [(15, 7), (23, 5), (23, 0)]:
        A = np.load(f'{d}/attn_{L}.npy', mmap_mode='r'); s = []
        for b, (n, on_, ids) in enumerate(R):
            Ab = np.asarray(A[b, h, :n, :n], np.float32); prev = np.concatenate([[-1], ids[:-1]])
            sink = Ab[:, 0]
            ind = np.array([Ab[q, :q + 1][prev[:q + 1] == ids[q]].sum() for q in range(n)])
            pre = np.arange(100, on_); lp_ = np.arange(on_ + 10, n)
            s.append((sink[pre].mean(), sink[lp_].mean(), ind[pre].mean(), ind[lp_].mean(), np.mean(ind[pre] > 0.5), np.mean(ind[lp_] > 0.5)))
        a = np.mean(s, 0)
        print(f'{d[-6:]} {L}.{h}: sink pre {a[0]:.2f} loop {a[1]:.2f} | ind pre {a[2]:.2f} loop {a[3]:.2f} | share ind>0.5 pre {a[4]:.2f} loop {a[5]:.2f}  (onsets {[r[1] for r in R]})')
```