The final norm weight has mean 3.3 and the layer-15 input norm has mean 0.2. That indicates zero-centred norm weights applied as (1 + w), the Qwen3-Next convention, so I'll use (1 + w) when attributing. Now the direct-logit attribution on the two Edit-A loop windows. For each layer I split the recorded residual change into the MLP output (down-projection applied to the recorded activations) and the mixer output (the rest). I score each against the actual next token's centred unembedding, divided by the final residual's RMS. I compare positions inside the exact loop with pre-loop positions. My earlier analysis of the original model found the in-loop excess came almost entirely from layers 15 and 23. If, under Edit A, the excess now comes mostly from layer 19, from other heads of layer 23, or from DeltaNet mixers, that tells me where copying has moved. If the excess is small and spread out, there is no single remaining copy route to cut.

```python
# Hypothesis: identify which layer components (mixer vs MLP) supply the in-loop next-token logit excess under edit A
E_mean = Eemb.mean(0)
H11 = [np.load(f'/work/fwd_11/hidden_{l}.npy', mmap_mode='r') for l in range(25)]
dla_res = {}
for b in range(2):
    x = ids11[b,:lens11[b]]; n_=len(x); o=ons[b]; Pc=per[b]
    t_in = np.arange(o+Pc, n_-1); t_pre = np.arange(64, o-1)
    tsel = np.concatenate([t_pre, t_in]); y = x[tsel+1]
    h24 = np.asarray(H11[24][b, tsel]).astype(np.float64)
    rms = np.sqrt((h24**2).mean(1) + 1e-6)
    U_ = (Eemb[y] - E_mean) * (1.0 + wn.numpy())[None,:] / rms[:,None]   # (T,1024)
    out = []
    for l in range(1, 25):
        dl = np.asarray(H11[l][b, tsel]) - np.asarray(H11[l-1][b, tsel])
        act = np.load(f'/work/fwd_11/mlp_act_{l-1}.npy', mmap_mode='r')[b, tsel]
        Wd_ = load(f'model.language_model.layers.{l-1}.mlp.down_proj.weight').numpy()
        mo = np.asarray(act) @ Wd_.T
        mix = dl - mo
        c_mlp = (mo*U_).sum(1); c_mix = (mix*U_).sum(1)
        npre = len(t_pre)
        out.append((l-1, c_mix[npre:].mean()-c_mix[:npre].mean(), c_mlp[npre:].mean()-c_mlp[:npre].mean(), c_mix[npre:].mean(), c_mlp[npre:].mean()))
    dla_res[b] = out
print('layer | mixer excess w0 w1 | mlp excess w0 w1')
for i in range(24):
    a0_, a1_ = dla_res[0][i], dla_res[1][i]
    print('%2d  mix %+5.2f %+5.2f | mlp %+5.2f %+5.2f' % (a0_[0], a0_[1], a1_[1], a0_[2], a1_[2]))
print('totals w0 mix %.2f mlp %.2f | w1 mix %.2f mlp %.2f' % (sum(r[1] for r in dla_res[0]), sum(r[2] for r in dla_res[0]), sum(r[1] for r in dla_res[1]), sum(r[2] for r in dla_res[1])))
```