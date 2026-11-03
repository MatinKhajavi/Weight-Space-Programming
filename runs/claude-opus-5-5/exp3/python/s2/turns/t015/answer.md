```python
import numpy as np
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def cs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),24)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0]),30)
O5=cs(OV0(5),22); O1=cs(OV0(1),22); O4=cs(OV0(4),3)
F=W['W_fc.0']*w2
tot=np.linalg.norm(F,axis=1)**2
for nm,S in [('T',T),('P',P),('O5',O5),('O1',O1),('O4',O4)]:
    fr=np.linalg.norm(F@S,axis=1)**2/tot
    print(nm, 'mean frac',fr.mean().round(3), 'quantiles',np.round(np.quantile(fr,[0.1,0.5,0.9,0.99]),3))
print('F sv',np.round(np.linalg.svd(F,compute_uv=False)[:40],1))
```