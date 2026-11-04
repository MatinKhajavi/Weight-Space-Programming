The paired generation failed twice in the same way at engine start-up, even after I removed the MLP activation recording. Both attempts were the first passes ever run with an edit present: pass 1 used only the original weights and pass 2 was teacher-forced on the original. So the edited tensor files themselves may be the problem, for example a wrong dtype, file layout or name that the server cannot load. Before trying again I inspect what `save_edit` wrote to `/work/edited`: the file names, tensor names, shapes, dtypes and any non-finite values. If the files look malformed I will rewrite the edit in the original dtype. If they look fine, the failure is probably on the server side and I will simply retry later, perhaps with a plain request without activations.

```python
# Diagnose engine-init failure: inspect the saved edit files (names, dtype, shape, finiteness) vs original
import glob
from safetensors import safe_open
print(sorted(glob.glob('/work/edited/**', recursive=True))[:20])
for f in glob.glob('/work/edited/**/*.safetensors', recursive=True):
    with safe_open(f, 'pt') as fh:
        for k in fh.keys():
            t = fh.get_tensor(k); print(f, k, tuple(t.shape), t.dtype, 'finite', bool(torch.isfinite(t.float()).all()))
orig_files = sorted(glob.glob('/model/*.safetensors')); print(orig_files[:5])
with safe_open(orig_files[0], 'pt') as fh:
    k0 = [k for k in fh.keys() if 'layers.16.mlp.down_proj' in k]
    print('orig dtype example:', [ (k, fh.get_slice(k).get_dtype()) for k in k0][:2], 'n keys', len(list(fh.keys())))
```