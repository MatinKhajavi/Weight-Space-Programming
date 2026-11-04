```python
# Weight anomaly scan: per-head norms in full-attn layers, per-head in DeltaNet, norm weights
import torch, numpy as np
P = 'model.language_model.layers.'
full_layers = [3,7,11,15,19,23]
for L in full_layers:
    q = load(P+f'{L}.self_attn.q_proj.weight'); k = load(P+f'{L}.self_attn.k_proj.weight')
    v = load(P+f'{L}.self_attn.v_proj.weight'); o = load(P+f'{L}.self_attn.o_proj.weight')
    qn = load(P+f'{L}.self_attn.q_norm.weight'); kn = load(P+f'{L}.self_attn.k_norm.weight')
    if L==3: print('shapes', q.shape, k.shape, v.shape, o.shape, qn.shape, kn.shape)
    oh = o.reshape(1024, 8, 256).norm(dim=(0,2))
    vh = v.reshape(2,256,1024).norm(dim=(1,2))
    kh = k.reshape(2,256,1024).norm(dim=(1,2))
    qh = q.reshape(8,512,1024)
    qq = qh[:, :256].norm(dim=(1,2)); qg = qh[:, 256:].norm(dim=(1,2))
    print(L, 'o', np.round(oh.numpy(),1), 'v', np.round(vh.numpy(),1), 'k', np.round(kh.numpy(),1))
    print('   q', np.round(qq.numpy(),1), 'gate', np.round(qg.numpy(),1), 'qn', round(qn.mean().item(),2), round(qn.std().item(),2), 'kn', round(kn.mean().item(),2), round(kn.std().item(),2))
```