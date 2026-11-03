```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
g=W['ln1_w.0']; b=W['ln1_b.0']
def sub(j):
    s=slice(64*j,64*j+64)
    Wq=W['W_q.0'][s]*g; Wk=W['W_k.0'][s]*g; Wv=W['W_v.0'][s]*g; Wo=W['W_o.0'][:,s]
    return Wq,Wk,Wv,Wo
Wq,Wk,Wv,Wo=sub(2)
QK=Wq.T@Wk
u,sv,vt=np.linalg.svd(QK)
print('L0H2 QK sv',sv[:30])
Pq=u[:,:20]; Pk=vt[:20].T
for j in [0,1,4,5,3]:
    _,_,Wv_,Wo_=sub(j)
    OV=Wo_@Wv_
    _,s2,vt2=np.linalg.svd(OV)
    for r in range(3):
        v=vt2[r]
        print(j,r,'sv %.2f  proj onto H2 q-space %.3f k-space %.3f'%(s2[r],np.linalg.norm(Pq.T@v),np.linalg.norm(Pk.T@v)))
# L0H2 value input: overlap with its own key/query subspace
_,s2,vt2=np.linalg.svd(Wo@Wv)
print('H2 OV sv',s2[:20])
print('H2 OV input top10 dirs proj onto H2 k-space', [round(float(np.linalg.norm(Pk.T@vt2[r])),3) for r in range(10)])
```