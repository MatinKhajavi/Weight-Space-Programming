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
S={}
S['L0H2']=outsp(0,2,20)
S['L0broad']=orth(np.concatenate([outsp(0,j,3) for j in [0,1,4,5]],1),3)
S['L0H3']=outsp(0,3,8)
for l in [1,2]:
    for j in range(6): S[f'L{l}H{j}']=outsp(l,j,20)
for l in range(3):
    S[f'MLP{l}']=orth(W[f'W_proj.{l}'],40)
names=list(S.keys())
print('writer dims',[S[n].shape[1] for n in names])
for l in [1,2,3]:
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j); A=Wq.T@Wk; n2=np.linalg.norm(A)**2
        kf=[np.linalg.norm(A@S[n])**2/n2/(S[n].shape[1]/384) for n in names if int(n[1])<l if True]
        qf=[np.linalg.norm(S[n].T@A)**2/n2/(S[n].shape[1]/384) for n in names if int(n[1])<l]
        vf=[np.linalg.norm((Wo@Wv)@S[n])**2/np.linalg.norm(Wo@Wv)**2/(S[n].shape[1]/384) for n in names if int(n[1])<l]
        nm=[n for n in names if int(n[1])<l]
        print(f'L{l}H{j} K:',' '.join(f'{a}={b:.1f}' for a,b in zip(nm,kf)))
        print(f'      Q:',' '.join(f'{a}={b:.1f}' for a,b in zip(nm,qf)))
        print(f'      V:',' '.join(f'{a}={b:.1f}' for a,b in zip(nm,vf)))
```