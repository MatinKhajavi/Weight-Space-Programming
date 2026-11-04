The picture is more interesting than a simple brake. Neuron 22.573's output direction is a "wait" direction: its unembedding projection puts " wait" first of all 248k tokens, "Wait" eighth and " Wait" eleventh (z-scores 19, 15 and 14), "Hmm" 181st, while "So", "Okay" and "$" sit near zero. At line starts its activation is negative, so it subtracts from the "Wait" logit. It is most negative where a "Wait" follows (−4.0 / −3.6) and less negative at other line starts (−2.4 / −1.7). So the neuron counterbalances the "Wait" impulse, cutting about 4 logits exactly where the rest of the network pushes for "Wait". It behaves similarly before and inside the loop. Amplifying its output would strengthen this built-in counterweight. Because the down-projection column scales both signs, I first need to know how the activation is distributed overall. If it is positive in some contexts, amplifying it would raise "wait" there and could backfire. I'll print activation quantiles across both windows and the contexts of its most positive activations.

```python
# Check sign profile of neuron 22.573: does it ever fire positive (promoting 'wait'), and where?
acts573 = np.concatenate([np.asarray(np.load('/work/fwd_11/mlp_act_22.npy', mmap_mode='r')[b, :lens11[b]-1, 573]) for b in range(2)])
print('quantiles', np.round(np.quantile(acts573, [0, .01, .05, .25, .5, .75, .95, .99, 1]), 2))
for b in range(2):
    x = ids11[b,:lens11[b]]
    a = np.asarray(np.load('/work/fwd_11/mlp_act_22.npy', mmap_mode='r')[b, :lens11[b]-1, 573])
    top = np.argsort(a)[::-1][:4]
    for t in top:
        ctx = tok.decode([int(i) for i in x[max(0,t-8):t+1]]).encode('ascii','replace').decode()
        print('win%d t=%d act %+.2f ctx=%r -> next %r' % (b, t, a[t], ctx[-50:], ts(x[t+1])))
# also original-model probe texts (fwd_4) if available
import glob
f4 = sorted(glob.glob('/work/fwd_4/mlp_act_22.npy')); print(f4)
```