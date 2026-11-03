```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
Q,K,V,O=headmats(0,3); OV3=O@V
U,S,Vt=np.linalg.svd(OV3); Prev=U[:,:20]; TokL=Vt[:20].T
U2,S2,Vt2=np.linalg.svd(Q.T@K); PosL=np.linalg.qr(np.hstack([U2[:,:20],Vt2[:20].T]))[0][:,:24]
Tokraw=np.linalg.qr(TokL/lw0[:,None])[0]; Posraw=np.linalg.qr(PosL/lw0[:,None])[0]
Q0,K0,V0,O0=headmats(0,0); Ul,Sl,_=np.linalg.svd(O0@V0); Loc=Ul[:,:10]
outs=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); outs.append(np.linalg.svd(o@v)[0][:,:2])
Pout=np.linalg.svd(np.hstack(outs))[0][:,:2]
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1); act=nf>1.0
Fa=F[act]
for name,B in [('tok',Tokraw),('pos',Posraw),('prev',Prev),('loc',Loc),('Pout',Pout)]:
    fr=(np.linalg.norm(Fa@B,axis=1)/np.linalg.norm(Fa,axis=1))**2
    print(name,'mean energy frac %.3f'%fr.mean(),'random exp %.3f'%(B.shape[1]/384), 'pct',np.percentile(fr,[10,50,90]).round(2))
# overlaps between subspaces
for a,A in [('tok',Tokraw),('prev',Prev),('loc',Loc),('pos',Posraw)]:
    print(a,[np.linalg.svd(A.T@B,compute_uv=False)[:3].round(2) for B in [Tokraw,Prev,Loc,Posraw,Pout]])
```