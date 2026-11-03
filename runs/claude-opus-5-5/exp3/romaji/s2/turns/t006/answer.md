```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
b=W['ln1_b.0']
for h in range(6):
    d=H[0][h]
    U,S,Vt=np.linalg.svd(d['QK']/8)
    s=slice(64*h,64*h+64)
    qc=W['W_q.0'][s]@b; kc=W['W_k.0'][s]@b
    keypref=d['K'].T@qc/8
    qpref=d['Q'].T@kc/8
    print(h,'|qc|=%.3f |kc|=%.3f |keypref|=%.3f |qpref|=%.3f'%(np.linalg.norm(qc),np.linalg.norm(kc),np.linalg.norm(keypref),np.linalg.norm(qpref)),
          'cos(keypref,v1)=%.3f'%(keypref@Vt[0]/np.linalg.norm(keypref)), 'S1=%.3f'%S[0])
outs={}
for h in range(6):
    U,S,Vt=np.linalg.svd(H[0][h]['OV']); outs[h]=(U[:,:3],Vt[:3])
for a in range(6):
    print(a,[np.round(np.linalg.svd(outs[a][0].T@outs[c][0],compute_uv=False),2) for c in range(6)])
print('input sides')
for a in range(6):
    print(a,[np.round(np.linalg.svd(outs[a][1]@outs[c][1].T,compute_uv=False),2) for c in range(6)])
```