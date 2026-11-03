```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1)
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)])
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
Fa=F[nf>1.0]; Fr=Fa-(Fa@Hout)@Hout.T
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:30].T
q,k,v,o=headmats(0,3); OV3=o@v
Q3,K3,V3,O3=headmats(3,3); QK=Q3.T@K3/8
M=D.T@QK@OV3@D
ev=np.linalg.eigvals(M[:26,:26]); ev=ev[np.argsort(-ev.real)]
print('eig M26',ev.round(3))
Ms=(M[:26,:26]+M[:26,:26].T)/2; print('sym eig',np.sort(np.linalg.eigvalsh(Ms))[::-1].round(3))
print('diag of M30 in D basis',np.diag(M).round(3))
```