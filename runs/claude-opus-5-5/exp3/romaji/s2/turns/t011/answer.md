```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K/8,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
def sv(A): return np.linalg.svd(A)
U,S,Vt=sv(H[0][3]['OV']); Tok=Vt[:20].T
U,S,Vt=sv(H[0][3]['QK']); Pos=np.linalg.qr(np.hstack([U[:,:20],Vt[:20].T]))[0][:,:24]
M=np.hstack([sv(H[0][h]['OV'])[2][:2].T for h in [1,2,4,5]])
Pin=sv(M)[0][:,:2]
print('Tok vs Pos',np.linalg.svd(Tok.T@Pos,compute_uv=False).round(2))
print('Pin vs Tok',np.linalg.svd(Pin.T@Tok,compute_uv=False).round(2))
print('Pin vs Pos',np.linalg.svd(Pin.T@Pos,compute_uv=False).round(2))
for h in [1,2,4,5,0]:
    U,S,Vt=sv(H[0][h]['QK'])
    print(h,'query u1: Tok %.2f Pos %.2f Pin %.2f | key v1: Tok %.2f Pos %.2f Pin %.2f'%(
        np.linalg.norm(Tok.T@U[:,0]),np.linalg.norm(Pos.T@U[:,0]),np.linalg.norm(Pin.T@U[:,0]),
        np.linalg.norm(Tok.T@Vt[0]),np.linalg.norm(Pos.T@Vt[0]),np.linalg.norm(Pin.T@Vt[0])))
    Uo,So,Vo=sv(H[0][h]['OV'])
    print('   OV in(top3): Tok',np.linalg.svd(Vo[:3]@Tok,compute_uv=False).round(2),'Pos',np.linalg.svd(Vo[:3]@Pos,compute_uv=False).round(2))
```