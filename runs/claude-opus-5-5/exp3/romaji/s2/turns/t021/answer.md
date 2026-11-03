```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
P3=W['W_proj.3'].astype(np.float64)
rn=np.linalg.norm(P3,axis=1); 
order=np.argsort(-rn); print('row norms top',order[:15],rn[order[:15]].round(2),'median',np.median(rn).round(2))
# find special dims via ln biases magnitude across layers
spec=set()
for k in ['ln1_b.1','ln1_b.2','ln1_b.3','ln2_b.0','ln2_b.1','ln2_b.2','ln2_b.3']:
    a=np.abs(W[k]); spec|=set(np.argsort(-a)[:6].tolist())
spec=sorted(spec); print('special dims',spec)
mask=np.ones(384,bool); mask[spec]=False
Pm=P3[mask]
U,S,Vt=np.linalg.svd(Pm); print('P3 without special dims',S[:40].round(2))
ipr=(U**4).sum(0); print('ipr',ipr[:40].round(3))
# L3H3 output subspace overlap with P3 top-k
lw=W['ln1_w.3']; s=slice(192,256)
OV=W['W_o.3'][:,s]@(W['W_v.3'][s]*lw); OVm=OV[mask]
Uo,So,_=np.linalg.svd(OVm)
for k in [10,15,18,20,21,22,23,24,25,26,27,28,30,34]:
    e=np.linalg.norm(U[:,:k].T@OVm)**2/np.linalg.norm(OVm)**2
    print(k,'L3H3 OV energy in top-k of P3: %.3f'%e, ' per-dir increment %.3f'%(np.linalg.norm(U[:,k-1].T@OVm)**2/np.linalg.norm(OVm)**2*mask.sum()))
```