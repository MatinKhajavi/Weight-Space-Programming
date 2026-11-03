```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
def outsp(l,j,r):
    Wq,Wk,Wv,Wo=headmats(l,j); u=np.linalg.svd(Wo@Wv)[0]; return u[:,:r]
def orth(M,r=None):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u if r is None else u[:,:r]
S={};LY={}
S['L0H2']=outsp(0,2,20);LY['L0H2']=0
S['L0brd']=orth(np.concatenate([outsp(0,j,3) for j in [0,1,4,5]],1),3);LY['L0brd']=0
S['L0H3']=outsp(0,3,8);LY['L0H3']=0
S['M0']=orth(W['W_proj.0'],40);LY['M0']=0
for l in [1,2]:
    for j in range(6): S[f'L{l}H{j}']=outsp(l,j,20);LY[f'L{l}H{j}']=l
    S[f'M{l}']=orth(W[f'W_proj.{l}'],40);LY[f'M{l}']=l
names=list(S.keys())
for l in [1,2,3]:
    nm=[n for n in names if LY[n]<l]
    print('reader layer',l,' cols:',' '.join(f'{n:>5}' for n in nm))
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j); A=Wq.T@Wk; n2=np.linalg.norm(A)**2; OV=Wo@Wv
        kf=[np.linalg.norm(A@S[n])**2/n2/(S[n].shape[1]/384) for n in nm]
        qf=[np.linalg.norm(S[n].T@A)**2/n2/(S[n].shape[1]/384) for n in nm]
        vf=[np.linalg.norm(OV@S[n])**2/np.linalg.norm(OV)**2/(S[n].shape[1]/384) for n in nm]
        print(f'L{l}H{j} K:',' '.join(f'{b:5.1f}' for b in kf))
        print(f'      Q:',' '.join(f'{b:5.1f}' for b in qf))
        print(f'      V:',' '.join(f'{b:5.1f}' for b in vf))
```