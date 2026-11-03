```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1)
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)])
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
Fa=F[nf>1.0]; Fr=Fa-(Fa@Hout)@Hout.T
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:26].T
q,k,v,o=headmats(0,3); Prev=np.linalg.svd(o@v)[0][:,:20]
pouts=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); pouts.append(np.linalg.svd(o@v)[0][:,:2])
Pout=np.linalg.svd(np.hstack(pouts))[0][:,:2]
for l in range(4):
    b=W[f'ln1_b.{l}']
    for h in range(6):
        q,k,v,o=headmats(l,h); s=slice(64*h,64*h+64)
        qc=W[f'W_q.{l}'][s]@b; kp=k.T@qc/8
        n=np.linalg.norm(kp)
        print(l,h,'|keypref| %.2f'%n,'frac tok(D) %.2f prev %.2f Pout %.2f'%(np.linalg.norm(D.T@kp)**2/n**2,np.linalg.norm(Prev.T@kp)**2/n**2,np.linalg.norm(Pout.T@kp)**2/n**2),
              ' D-coeffs',(D.T@kp).round(2)[:26])
```