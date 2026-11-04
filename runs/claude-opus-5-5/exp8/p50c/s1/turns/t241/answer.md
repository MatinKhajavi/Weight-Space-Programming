The last run failed with a bookkeeping error (`KeyError: 21`): a dictionary keyed by pass number had no entry for 21. It was not a problem with the model or the data. Before going on, I need to re-establish the state. First, what does `/work/gen_22.jsonl` contain? Per the notes it should be pass 17, X1 = C plus zeroing head 23.4, on the 512 S1+H1 items. Second, which heads and columns are actually edited in the current weights? I will also list the helper functions still defined in the kernel so I can reuse them. If gen_22 is X1 and head 23.4's o_proj columns are zeroed now, the next step is the paired comparison of X1 against C (gen_13 and gen_16_edited) on the same items.

```python
# State check: what is gen_22 (expected: X1 = C + head 23.4 zeroed, on SH512) and what is currently edited?
import json, numpy as np, torch, types
def readj(p): return [json.loads(l) for l in open(p)]
g22 = readj('/work/gen_22.jsonl')
r0 = g22[0]
print(len(g22), {k:(v if not isinstance(v,str) else v[:40]) for k,v in r0.items() if not (isinstance(v,str) and len(v)>200)})
for L in (15,23):
    W = load(f'model.language_model.layers.{L}.self_attn.o_proj.weight'); W0 = load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    print(f'L{L} head norm ratio:', [round(float(W[:,h*256:(h+1)*256].norm()/W0[:,h*256:(h+1)*256].norm()),2) for h in range(8)])
D = load('model.language_model.layers.22.mlp.down_proj.weight'); D0 = load_original('model.language_model.layers.22.mlp.down_proj.weight')
dif = (D-D0).norm(dim=0); idx = torch.nonzero(dif>1e-6).flatten().tolist()
print('L22 down changed cols:', idx, [round(float(D[:,i].norm()/D0[:,i].norm()),2) for i in idx])
print('helpers:', sorted([k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')]))
print(open('/work/notes.md').read()[-600:])
```