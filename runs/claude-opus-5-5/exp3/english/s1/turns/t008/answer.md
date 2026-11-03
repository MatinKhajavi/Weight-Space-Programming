```python
import numpy as np
def edge(m,n,s=0.02): return s*(np.sqrt(m)+np.sqrt(n))
for l in range(4):
    out=[]
    for nm in ['W_q','W_k','W_v']:
        for h in range(6):
            M=W[f'{nm}.{l}'][64*h:64*h+64]
            sv=np.linalg.svd(M,compute_uv=False)
            out.append(f'{nm[-1]}{h}:{np.sum(sv>edge(64,384)*1.15)}')
    M=W[f'W_o.{l}']
    for h in range(6):
        sv=np.linalg.svd(M[:,64*h:64*h+64],compute_uv=False)
        out.append(f'o{h}:{np.sum(sv>edge(64,384,0.00707)*1.15)}')
    sv=np.linalg.svd(W[f'W_fc.{l}'],compute_uv=False); out.append(f'fc:{np.sum(sv>edge(1536,384)*1.15)}')
    sv=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False); out.append(f'proj:{np.sum(sv>edge(1536,384,0.00707)*1.15)}')
    print('L',l,' '.join(out))
for nm in ['W_fc.0','W_proj.0','W_proj.3','W_o.3','W_fc.3']:
    print(nm, np.round(np.linalg.svd(W[nm],compute_uv=False)[:80],2))
```