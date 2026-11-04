The output was swamped because each record nests its transcript in `samples[0]['text']`, which my filter for long strings did not catch, so the edit-state lines were cut off. I now know the record structure: `id`, `source`, `answer`, and `samples`, a list of dicts holding the text and probably the grade. I will rerun the state check and print only the scalar fields of a sample. If gen_22 is X1 (C plus head 23.4 zeroed), its accuracy should be compared paired with C on the same seeds. C's results are in gen_13 for the S1 items and gen_16_edited for the H1 items.

```python
# State check v2: sample fields of gen_22 (no text), current edit state, helper functions
s0 = g22[0]['samples'][0]
print(len(g22), 'sample fields:', {k:(v if not isinstance(v,str) else v[:30]) for k,v in s0.items() if k!='text'})
for L in (15,23):
    W = load(f'model.language_model.layers.{L}.self_attn.o_proj.weight'); W0 = load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    print(f'L{L} head norm ratio:', [round(float(W[:,h*256:(h+1)*256].norm()/W0[:,h*256:(h+1)*256].norm()),2) for h in range(8)])
D = load('model.language_model.layers.22.mlp.down_proj.weight'); D0 = load_original('model.language_model.layers.22.mlp.down_proj.weight')
dif = (D-D0).norm(dim=0); idx = torch.nonzero(dif>1e-6).flatten().tolist()
print('L22 down changed cols:', idx, [round(float(D[:,i].norm()/D0[:,i].norm()),2) for i in idx])
print('helpers:', sorted([k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')])[:80])
```