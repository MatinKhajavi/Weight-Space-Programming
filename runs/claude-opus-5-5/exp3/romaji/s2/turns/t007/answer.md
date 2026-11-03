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
# shared input subspace from heads 1,2,4,5 OV
M=np.hstack([np.linalg.svd(H[0][h]['OV'])[2][:2].T for h in [1,2,4,5]])
Uin,Sin,_=np.linalg.svd(M); Pin=Uin[:,:2]; print('Sin',Sin.round(3))
M=np.hstack([np.linalg.svd(H[0][h]['OV'])[0][:,:2] for h in [1,2,4,5]])
Uout,Sout,_=np.linalg.svd(M); Pout=Uout[:,:2]; print('Sout',Sout.round(3))
for h in range(6):
    U,S,Vt=np.linalg.svd(H[0][h]['QK']/8)
    print(h,'S',S[:4].round(3),'key v1 in Pin: %.3f'%np.linalg.norm(Pin.T@Vt[0]),'query u1 in Pin: %.3f'%np.linalg.norm(Pin.T@U[:,0]),
          'v1 in Pout %.3f'%np.linalg.norm(Pout.T@Vt[0]), 'u1.v1 %.3f'%(U[:,0]@Vt[0]))
print('Pin vs Pout', np.linalg.svd(Pin.T@Pout,compute_uv=False))
# Input subspace and LN weights: which residual dims dominate?
lw=W['ln1_w.0']
for i in range(2):
    v=Pin[:,i]; idx=np.argsort(-np.abs(v))[:10]; print('Pin',i,idx, v[idx].round(3), 'ipr',(v**4).sum()/(v**2).sum()**2)
for i in range(2):
    v=Pout[:,i]; idx=np.argsort(-np.abs(v))[:10]; print('Pout',i,idx, v[idx].round(3),'ipr',(v**4).sum()/(v**2).sum()**2)
```