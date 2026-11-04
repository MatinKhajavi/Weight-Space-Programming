The projection works as intended. The protected column keeps about 91% of the "Wait" effect (0.741 vs 0.813 logits per unit activation) and most of "Hmm"/"Actually". Its effect on " Ready", "</think>" and " Final" becomes exactly zero, while the column direction barely changes (cosine 0.99). For a clean test I will rescale the protected column so its "Wait" suppression equals edit C's (factor 2.194, giving 2×0.813). V2 then differs from C only in no longer suppressing the three concluding tokens. At an activation of -4 that is about +1.0 logit for "</think>" and +1.8 for " Ready" relative to C, exactly where the neuron fires: line starts that open a new thought. I will save V2 and check how many unused held-out problems remain for a later fresh confirmation set. Then I will run V2 on S1+H1 (512 items) paired with the existing C runs on the same seeds. If V2 gains accuracy mainly through fewer cut answers without many more wrong answers, the brake's off-target suppression of concluding was a real cause of failing to conclude.

```python
# Build V2: protected brake column (no effect on ' Ready', '</think>', ' Final'), rescaled so Wait suppression equals C; check remaining pool sizes
name22='model.language_model.layers.22.mlp.down_proj.weight'
kV2=float(2*eff_tokens(c573)[' Wait']/eff_tokens(c_prot)[' Wait'])
W22v2=W22o.clone(); W22v2[:,573]=kV2*c_prot
save_edit(name22, W22v2)
print('kV2', round(kV2,3), 'edited:', edited())
print('V2 col effects:', eff_tokens(load(name22)[:,573]))
pools_=json.load(open('/work/pools.json')); print({k:(len(v) if hasattr(v,'__len__') else v) for k,v in pools_.items()})
S1ids=json.load(open('/work/S1_ids.json')); H1ids=json.load(open('/work/H1_ids.json')); print(len(S1ids), len(H1ids), S1ids[:2], H1ids[:2])
```