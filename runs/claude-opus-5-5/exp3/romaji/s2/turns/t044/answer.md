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
q,k,v,o=headmats(0,3); Prev=np.linalg.svd(o@v)[0][:,:20]; U2,S2,V2=np.linalg.svd(q.T@k); Pos=np.linalg.qr(np.hstack([U2[:,:12],V2[:12].T]))[0][:,:20]
q,k,v,o=headmats(0,0); Loc=np.linalg.svd(o@v)[0][:,:10]
pouts=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); pouts.append(np.linalg.svd(o@v)[0][:,:2])
Pout=np.linalg.svd(np.hstack(pouts))[0][:,:2]
subs={'D':D,'Prev':Prev,'Pos':Pos,'Loc':Loc,'Pout':Pout}
print('per-dim normalized key-side read  ||QK B||^2/(dim B) / (||QK||^2/384)')
for l in range(4):
    for h in range(6):
        q,k,v,o=headmats(l,h); QK=q.T@k; tot=np.linalg.norm(QK)**2/384
        ks={n:np.linalg.norm(QK@B)**2/B.shape[1]/tot for n,B in subs.items()}
        qs={n:np.linalg.norm(B.T@QK)**2/B.shape[1]/tot for n,B in subs.items()}
        vv=v.T; OVin={n:np.linalg.norm((o@v)@B)**2/B.shape[1]/(np.linalg.norm(o@v)**2/384) for n,B in subs.items()}
        print(l,h,'K:',' '.join(f'{n}={x:.1f}' for n,x in ks.items()),'| Q:',' '.join(f'{n}={x:.1f}' for n,x in qs.items()),'| OVin:',' '.join(f'{n}={x:.1f}' for n,x in OVin.items()))
```