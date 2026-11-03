```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
M=np.hstack([sv(H[0][h]['OV'])[0][:,:2] for h in [1,2,4,5]]); Pout=sv(M)[0][:,:2]
for l in [1,2,3]:
    b=W[f'ln1_b.{l}']; lw=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        U,S,Vt=sv(H[l][h]['QK'])
        qc=W[f'W_q.{l}'][s]@b; kpref=H[l][h]['K'].T@qc/8
        kc=W[f'W_k.{l}'][s]@b; qpref=H[l][h]['Q'].T@kc/8
        print(l,h,'S',S[:3].round(2),'v1 in Pout %.2f u1 in Pout %.2f'%(np.linalg.norm(Pout.T@Vt[0]),np.linalg.norm(Pout.T@U[:,0])),
              '|keypref| %.2f  keypref in Pout %.2f  cos(kp,v1) %.2f'%(np.linalg.norm(kpref),np.linalg.norm(Pout.T@kpref)/np.linalg.norm(kpref),kpref@Vt[0]/np.linalg.norm(kpref)),
             '|qpref| %.2f'%np.linalg.norm(qpref))
# residual dims of Pout and the ln1_b, ln1_w values there
for l in range(4):
    print('layer',l,'ln1_b at 21,39,58,99:',W[f'ln1_b.{l}'][[21,39,58,99]].round(2),'ln1_w:',W[f'ln1_w.{l}'][[21,39,58,99]].round(2),'ln2_b',W[f'ln2_b.{l}'][[21,39,58,99]].round(2),'ln2_w',W[f'ln2_w.{l}'][[21,39,58,99]].round(2))
print(np.argsort(-np.abs(W['ln1_b.1']))[:10], np.sort(-np.abs(W['ln1_b.1']))[:10])
print(np.argsort(-np.abs(W['ln1_b.2']))[:10], np.sort(-np.abs(W['ln1_b.2']))[:10])
```