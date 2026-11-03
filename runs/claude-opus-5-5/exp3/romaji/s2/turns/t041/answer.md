```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
outs=[]; Ss=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); outs.append(U[:,:2]*S[:2]); Ss.append(S[:3])
Pout=np.linalg.svd(np.hstack(outs))[0][:,:2]
print('Pin head OV sv',[s.round(2) for s in Ss])
# typical magnitude of Pout component in residual: OV sv * |Pin component| ; unknown. report QK blocks
for l in [1,2,3]:
    b=W[f'ln1_b.{l}']; lw=W[f'ln1_w.{l}']
    for h in range(6):
        q,k,v,o=headmats(l,h); s=slice(64*h,64*h+64)
        QK=q.T@k/8
        B=Pout.T@QK@Pout
        qc=W[f'W_q.{l}'][s]@b; kp=(k.T@qc/8)@Pout
        print(l,h,'PoutxPout',B.round(3).tolist(),'eig',np.round(np.linalg.eigvals(B),3),'const-q key pref on Pout',kp.round(3))
```