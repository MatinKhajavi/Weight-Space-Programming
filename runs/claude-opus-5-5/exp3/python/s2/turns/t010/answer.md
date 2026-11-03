```python
import numpy as np
for l in range(4):
    w=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(64*j,64*j+64)
        M=(W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
        OV=W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*w)
        ev=np.linalg.eigvals(M); idx=np.argsort(-np.abs(ev))[:14]
        so=np.linalg.svd(OV,compute_uv=False)
        print(f'L{l}H{j} |QK|={np.linalg.norm(M):.1f} tr/fro={np.trace(M)/np.linalg.norm(M):+.2f} |OV|={np.linalg.norm(OV):.1f} OVsv={np.round(so[:4],1)} effrankOV={so.sum()**2/(so**2).sum():.1f}')
        print('    ', ' '.join(f'{abs(e):.1f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx]))
```