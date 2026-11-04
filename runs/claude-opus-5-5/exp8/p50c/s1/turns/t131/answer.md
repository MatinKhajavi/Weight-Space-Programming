The source isn't retrievable because the functions were defined interactively. I'll use their signatures and local variable names to recall what they do. I'll also check that the current edited tensors match A+brake x2 by comparing them with the originals.

```python
# Hypothesis: signatures suffice to reuse helpers; verify current edit = A (3 heads zeroed) + 22.573 x2
for fn in ['set_heads','snapshot','paired','readgen','dla','dla_diff','neuron_dla','indm','ind_targets','find_loop','wrate','summarize','prompt_of','tid','tokstr']:
    f=globals()[fn]
    try: print(fn, inspect.signature(f), f.__code__.co_varnames[:12])
    except Exception as e: print(fn,'ERR',e)
import torch
for n in edited():
    w=load(n); w0=load_original(n); d=(w-w0)
    nzc=(d.abs().sum(0)>0).nonzero().flatten()
    print(n, tuple(w.shape), 'changed cols', len(nzc), nzc[:3].tolist(), nzc[-3:].tolist())
w=load('model.language_model.layers.22.mlp.down_proj.weight'); w0=load_original('model.language_model.layers.22.mlp.down_proj.weight')
print('22.573 ratio', float((w[:,573]@w0[:,573])/(w0[:,573]@w0[:,573])))
```