```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
lw0=W['ln1_w.0']
L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=u[:,:28]
Ms={};names=[]
for l in range(4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        OV=W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*lw[None,:])
        Ms[f'L{l}H{j}']=U.T@OV@T; names.append(f'L{l}H{j}')
# also MLP3 "direct" map T->U: W_proj.3 @ W_fc.3 (linear approx ignoring relu) and W_proj.l for others
for l in range(4):
    Mlin = U.T@(W[f'W_proj.{l}']@(W[f'W_fc.{l}']*W[f'ln2_w.{l}'][None,:]))@T
    Ms[f'MLP{l}lin']=Mlin; names.append(f'MLP{l}lin')
n=len(names); C=np.zeros((n,n))
for a in range(n):
    for b in range(n):
        C[a,b]=np.sum(Ms[names[a]]*Ms[names[b]])/np.linalg.norm(Ms[names[a]])/np.linalg.norm(Ms[names[b]])
import pandas as pd
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(pd.DataFrame(C,index=names,columns=names).round(2).to_string())
print("norms of U^T OV T:", {k:round(float(np.linalg.norm(v)),2) for k,v in Ms.items()})
```