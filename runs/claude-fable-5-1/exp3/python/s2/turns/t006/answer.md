```python
np.set_printoptions(precision=2, suppress=True, linewidth=200)
hd=64
def subspace(M, energy=0.9):
    u,s,vt = np.linalg.svd(M, full_matrices=False)
    c = np.cumsum(s**2)/np.sum(s**2)
    r = int(np.searchsorted(c, energy))+1
    return vt[:r].T, r
def overlap(A,B):
    UA,ra = subspace(A); UB,rb = subspace(B)
    return (np.linalg.norm(UA.T@UB)**2)/min(ra,rb), ra, rb, max(ra,rb)/384
mats = {}
for l in range(4):
    lw = W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        for nm in 'qkv':
            mats[f'L{l}H{j}{nm}'] = W[f'W_{nm}.{l}'][s]*lw[None,:]
        mats[f'L{l}H{j}o'] = W[f'W_o.{l}'][:,s].T
names = [f'L0H{j}{nm}' for j in range(6) for nm in 'qkv']
print("effective ranks (90% energy) layer0:")
for n in names: print(n, subspace(mats[n])[1], end='; ')
print()
# overlap matrix among layer-0 q,k,v row spaces
print("overlap matrix (rows/cols: " + ",".join(names)+")")
M = np.zeros((len(names),len(names)))
for a,na in enumerate(names):
    for b,nb in enumerate(names):
        M[a,b]=overlap(mats[na],mats[nb])[0]
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(pd.DataFrame(M,index=names,columns=names).round(2))
```