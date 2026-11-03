```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def raw(l,j):
    s=slice(64*j,64*j+64); return W[f'W_q.{l}'][s],W[f'W_k.{l}'][s],W[f'W_v.{l}'][s],W[f'W_o.{l}'][:,s]
Wq,Wk,Wv2,Wo2=raw(0,2)
A2=Wq.T@Wk
u,s,vt=np.linalg.svd(A2)
P=np.linalg.qr(np.concatenate([u[:,:20],vt[:20].T],1))[0]
# add broad heads' value input dirs (top 3 each)
ext=[P]
for j in [0,1,4,5]:
    _,_,Wv,Wo=raw(0,j); _,ss,vv=np.linalg.svd(Wo@Wv); ext.append(vv[:3].T)
Pe=np.linalg.svd(np.concatenate(ext,1),full_matrices=False)[0]
k=np.sum(np.linalg.svd(np.concatenate(ext,1),compute_uv=False)>0.3)
Pe=Pe[:,:k]; print('Pext dims',k)
Pc=np.eye(384)-Pe@Pe.T
print('L0H2 Wv no Pext',np.linalg.svd(Wv2@Pc,compute_uv=False)[:36])
print('L0H2 OV no Pext',np.linalg.svd(Wo2@Wv2@Pc,compute_uv=False)[:36])
# W_v of H2 in raw y; how much energy in P vs Pext
print('energy Wv2 in P %.3f in Pext %.3f'%(np.linalg.norm(Wv2@P)**2/np.linalg.norm(Wv2)**2, np.linalg.norm(Wv2@Pe)**2/np.linalg.norm(Wv2)**2))
```