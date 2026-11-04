gen_21 is not edit C: its texts diverge from gen_13 after a median of about 850 characters. Runs of the same model are byte-identical (the two original runs gen_7 and gen_8 matched exactly), so gen_21 must have used a different set of weights. Its accuracy was the same as C (60.5% vs 60.2%; 18 problems went wrong→right and 17 right→wrong). Either an undocumented variant was tested and reverted, or the current /work/edited is not exactly C. I will compare the current edited tensors against the saved C snapshot and the original weights, to know exactly what is in /work/edited now.

```python
# Verify current /work/edited equals the saved C snapshot (best_C_x2) and characterize any difference from original
print(os.listdir('/work/edited'), os.listdir('/work/best_C_x2'))
from safetensors.torch import load_file
import torch
def snap(d):
    out = {}
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if f.endswith('.safetensors'):
            out.update(load_file(p))
    return out
bc = snap('/work/best_C_x2')
print('snapshot keys', list(bc.keys()))
for n in edited():
    cur = load(n); orig = load_original(n)
    dif = (cur - orig)
    nzc = (dif.abs().sum(0) > 0).nonzero().flatten() if dif.dim() == 2 else None
    print(n, 'changed cols', len(nzc), nzc[:12].tolist() if len(nzc) < 1000 else '', 'cur==snap', (n in bc) and torch.equal(cur, bc[n].float()))
```