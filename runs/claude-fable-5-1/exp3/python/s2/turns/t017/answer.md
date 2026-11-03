```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
lw0=W['ln1_w.0']
L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
vecs={}
for l in range(2):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        for nm in 'qk':
            M=W[f'W_{nm}.{l}'][sl]*lw[None,:]
            u,s,vt=np.linalg.svd(M@T,full_matrices=False)
            vecs[f'L{l}H{j}{nm}']=vt[0]
            print(f'L{l}H{j}{nm}: sv of K|T:', s[:6], " frac top1:", f"{s[0]**2/np.sum(s**2):.2f}")
names=list(vecs.keys())
C=np.array([[abs(vecs[a]@vecs[b]) for b in names] for a in names])
import pandas as pd
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(pd.DataFrame(C,index=names,columns=names).round(2).to_string())
```