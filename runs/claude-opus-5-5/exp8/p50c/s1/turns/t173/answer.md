The final norm uses its weights directly (mean 3.3), so the z-scores are valid. Neuron 22.573's column is a dedicated "Wait" output (z about 14 to 15), weaker on "Hmm" and "Actually", and essentially zero on "Maybe" (0.4 to 0.7). That explains why doubt reappears as "Maybe ..." lines. More importantly, the column is also positive on " Ready" (z 4.0) and "</think>" (z 2.2). Because the neuron fires negatively, our doubling also doubled its suppression of exactly the tokens that end a correct thinking phase ("Ready to generate output.\n</think>"). That could feed the failure-to-conclude cases. To judge whether this side effect matters, I will convert the column's effect into logits per unit of activation. I will use the real final-layer RMS from the C windows and check the neuron's typical negative activation there. If the extra suppression of </think> and " Ready" at negative activations is a sizeable fraction of a logit or more, a refined brake that doubles only the Wait component is a justified next edit.

```python
# Hypothesis: doubling 22.573 also suppresses concluding tokens (' Ready', '</think>') by a non-trivial number of logits at its typical negative activation
h24=np.load('/work/fwd_18/hidden_24.npy', mmap_mode='r'); h23=np.load('/work/fwd_18/hidden_23.npy', mmap_mode='r')
a22=np.load('/work/fwd_18/mlp_act_22.npy', mmap_mode='r')[:, :, 573]
L0,L1=m18['lengths']
rms24=np.concatenate([np.sqrt((np.asarray(h24[0,:L0])**2).mean(-1)), np.sqrt((np.asarray(h24[1,:L1])**2).mean(-1))])
print('final-layer residual RMS median', np.median(rms24).round(2), 'q10-90', np.percentile(rms24,[10,90]).round(2))
acts_=np.concatenate([a22[0,:L0], a22[1,:L1]])
print('22.573 act (C windows): median', np.median(acts_).round(2), 'q5', np.percentile(acts_,5).round(2), 'min', acts_.min().round(2), 'max', acts_.max().round(2))
c573=W22o[:,573]
eff={k:float((E_[v]-mu_E)@(g_*c573)) for k,v in fam.items() if v is not None}
rmsm=float(np.median(rms24))
print('logit per unit act (direct path, /median rms):', {k:round(v/rmsm,3) for k,v in eff.items() if k in [' Wait','Wait',' Hmm',' Actually',' Ready','</think>',' Maybe','Maybe',' But',' So',' Final']})
print('extra logit from doubling at act=-4:', {k:round(-4*v/rmsm,2) for k,v in eff.items() if k in [' Wait','Wait',' Ready','</think>',' Hmm',' Actually']})
```