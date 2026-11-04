The MLP screen gives small numbers: the total loop-specific excess from all MLP neurons in layers 12–23 is only 1.99 and 0.97 logits in the two windows. The top neurons (22.59, 19.11, 23.1149 and others) add at most about 0.2 each. So MLPs are not the main carriers of in-loop copying; attention is. But the list looks suspicious: far too many of the top neurons have small indices (2, 3, 7, 8, 11, 22, 26, 27, 29, 38, 59...). That should not happen with permutation-symmetric neurons, unless the activation export has an ordering or precision artifact or the model really concentrates large activations in low-index neurons. Before relying on any per-neuron result, including earlier ones from these files, I will check how activation magnitude and contribution are distributed over neuron indices. I will also verify that the stored activations for brake neuron 22.573 match what was seen earlier.

```python
# Sanity check: are low-index neurons artificially dominant in stored mlp_act (export artifact) or genuinely large?
for l_ in [19,22,23]:
    Aml=np.asarray(np.load(f'/work/fwd_18/mlp_act_{l_}.npy', mmap_mode='r')[0,:m18['lengths'][0]], dtype=np.float32)
    ma=np.abs(Aml).mean(0)
    print(l_, 'mean|act| idx<100', ma[:100].mean().round(4), '100-1000', ma[100:1000].mean().round(4), '>=1000', ma[1000:].mean().round(4),
          '| top-8 idx by mean|act|', np.argsort(-ma)[:8].tolist(), '| nonzero share', (np.abs(Aml)>1e-3).mean().round(3))
print('22.573 act text0: min', Aml.min() if False else np.asarray(np.load('/work/fwd_18/mlp_act_22.npy', mmap_mode='r')[0,:m18['lengths'][0],573],dtype=np.float32).min().round(2))
```