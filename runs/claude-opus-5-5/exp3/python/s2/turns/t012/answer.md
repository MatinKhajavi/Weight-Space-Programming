```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
def cs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
T=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),22)
subs={'T':T,'O5':cs(OV0(5),20),'O1':cs(OV0(1),20),'O4':cs(OV0(4),3),'O0':cs(OV0(0),2),'O2':cs(OV0(2),8),'O3':cs(OV0(3),8)}
# check overlaps
names=list(subs)
for a in names:
    print(a, ' '.join(f'{b}:{np.linalg.norm(subs[a].T@subs[b])**2/min(subs[a].shape[1],subs[b].shape[1]):.2f}' for b in names))
for l in [1,2,3]:
  for j in range(6):
    M=QK(l,j); tot=np.linalg.norm(M)**2
    res=[]
    for a in ['T','O5','O1','O4','O0']:
        for b in ['T','O5','O1','O4','O0']:
            e=np.linalg.norm(subs[a].T@M@subs[b])**2/tot
            if e>0.02: res.append(f'{a}x{b}:{e:.2f}')
    print(f'L{l}H{j}',' '.join(res))
```