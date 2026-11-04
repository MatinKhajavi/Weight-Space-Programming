Even under E4b, P(So) at a decision point still grows with the number of earlier identical lines: 0.12 at k=1 (the first repeat), then .32 / .43 / .54 / .61 / .68 for k=2..6. A repetition-driven feedback is still active for So, weaker than the original Wait loop (.57→.98) but enough to trap the model. Before attributing it to components, I need one technical check. Does hidden_24 hold the raw residual or the output of the final RMSNorm, as Hugging Face's last hidden state does? This decides how the DLA must be normalised. If it is post-norm, h24/gF has RMS exactly 1 at every position and its scale is far below h23's. I then also verify that the readout reproduces the returned top-1 token, and compute the per-layer DLA of the So margin by k.

```python
# Check hidden_24 convention (pre- vs post-final-norm), then DLA of So-margin by component and repetition k
H13 = {l: np.load(F13+f'hidden_{l}.npy', mmap_mode='r') for l in range(25)}
gFn_ = gF.numpy()
i0 = 0; Lx = lens13[i0]
h24_ = np.asarray(H13[24][i0, :Lx]); h23_ = np.asarray(H13[23][i0, :Lx])
r24 = np.sqrt((h24_**2).mean(-1)); r23 = np.sqrt((h23_**2).mean(-1))
rq = np.sqrt(((h24_/gFn_)**2).mean(-1))
print('rms h24 median', np.median(r24), ' rms h23 median', np.median(r23), ' rms(h24/gF) min/median/max', rq.min(), np.median(rq), rq.max())
# readout check on a few positions: pre-norm convention
WUn_ = WU.numpy() if isinstance(WU, torch.Tensor) else WU
pos_chk = [50, 100, 150]
for p in pos_chk:
    x = h24_[p]; xn = gFn_ * x / np.sqrt((x**2).mean() + 1e-6)
    lg_pre = WUn_ @ xn; lg_post = WUn_ @ x
    print(p, 'top1 returned', int(tki13[i0, p, 0]), 'pre-norm top1', int(lg_pre.argmax()), 'post-norm top1', int(lg_post.argmax()))
```