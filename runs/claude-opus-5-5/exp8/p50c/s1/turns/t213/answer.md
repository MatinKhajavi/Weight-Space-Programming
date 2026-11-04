Line-start retrieval in C's loops is entirely ordinary induction: the part not explained by induction is about 0 for every head. Among heads still active under C, 23.4 dominates it (0.86 of attention mass from line-start positions), followed by 11.1 (0.32). Head 19.0 is not a factor. However, 23.4 was part of edit B, where zeroing 23.4, 23.7, 19.5 and 19.7 on top of A did not reduce cut-offs at all (62.9% vs 61.3%) and quadrupled wrong answers. So the remaining copy heads look necessary for correct computation, and further head ablation is a poor bet. I will turn to the cleanest remaining side effect of our own edit. The doubled brake column also suppresses </think> (logit effect of +0.125 per unit of activation, applied at strongly negative activations). From recorded C windows (fwd_18), I will measure how often </think> is a live candidate at line starts and how much logit the extra brake copy removes there. If </think> is a frequent top-10 candidate whose suppression is a sizeable part of its margin, removing that component is worth a test.

```python
# How often is </think> a live candidate under C, and how much does the EXTRA 22.573 copy suppress it there? (DLA on recorded acts)
E = load('model.language_model.embed_tokens.weight'); nw = load('model.language_model.norm.weight')
d573 = load_original('model.language_model.layers.22.mlp.down_proj.weight')[:, 573]
def u(tid): return (1 + nw) * E[tid] if nw.mean() < 0.5 else nw * E[tid]
print('norm w mean', float(nw.mean()))
TH = 248069; WAIT_LI = 13784
for d in ['/work/fwd_18', '/work/fwd_11']:
    tk = np.load(d + '/topk_ids.npy'); tl = np.load(d + '/topk_logprobs.npy')
    h24 = np.load(d + '/hidden_24.npy', mmap_mode='r'); a22 = np.load(d + '/mlp_act_22.npy', mmap_mode='r')
    lens_ = json.load(open(d + '/meta.json'))['lengths']
    for b in range(2):
        n = lens_[b]; hit = np.where((tk[b, :n] == TH).any(-1))[0]
        rms = np.sqrt((np.asarray(h24[b, :n], np.float32) ** 2).mean(-1))
        act = np.asarray(a22[b, :n, 573], np.float32)
        contrib = act * float(d573 @ u(TH)) / rms   # contribution of ONE copy of the column to </think> logit
        lp = np.array([tl[b, p][tk[b, p] == TH][0] for p in hit]) if len(hit) else np.array([])
        print(d[-6:], b, 'n', n, '</think> in top10 at', len(hit), 'pos; lp quantiles', np.percentile(lp, [50, 90, 100]).round(2) if len(hit) else '-',
              '| one-copy contrib at those pos median', np.median(contrib[hit]).round(2) if len(hit) else '-', '| at all pos median', np.median(contrib).round(2), 'min', contrib.min().round(2))
```