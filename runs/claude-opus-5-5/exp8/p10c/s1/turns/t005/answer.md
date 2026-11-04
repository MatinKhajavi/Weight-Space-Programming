```python
# Hypothesis: get shapes of all tensor types for one linear-attn layer and one full-attn layer.
from safetensors import safe_open
f = safe_open('/model/model.safetensors', 'pt')
keys = list(f.keys())
print(len(keys))
for k in keys:
    if ('layers.0.' in k) or ('layers.3.' in k) or 'layers' not in k:
        sl = f.get_slice(k)
        print(k.replace('model.language_model.',''), sl.get_shape(), sl.get_dtype())
print([k for k in keys if 'mtp' in k][:10])
print([k for k in keys if 'visual' in k][:3])
```