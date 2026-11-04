The helper's source is not available. The architecture (Qwen3-Next lineage) uses zero-centred norm weights applied as (1 + w), and I will confirm this from the input-layernorm weight means: values near 0 mean zero-centred, near 1 mean plain. I also need the alignment of the top-k arrays, that is, whether index t holds the prediction for token t or for token t+1, so I check which convention matches the actual tokens inside the deterministic loop region.

Then I run the neuron attribution. For each neuron in layers 14–23, I take its average contribution to the logit of the copied token minus the best alternative over the 52 loop branch points, and over control line starts before the loop. Candidate brakes are neurons that are consistently negative at the branch points.

```python
# Neuron DLA at loop branch points vs pre-loop line starts (C model), layers 14-23
print('input_layernorm means:', [round(float(load_original(f'model.language_model.layers.{k}.input_layernorm.weight').mean()),2) for k in (0,7,15,23)])
gfz = 1+wf
tl18 = np.load(F+'token_logprobs.npy')
b0 = bps[5]; print('align check: topk[t,0]==ids[t]:', np.mean([tki[b,t,0]==ids18[b,t] for b,t in bps]), ' topk[t,0]==ids[t+1]:', np.mean([tki[b,t,0]==ids18[b,t+1] for b,t in bps]))
```