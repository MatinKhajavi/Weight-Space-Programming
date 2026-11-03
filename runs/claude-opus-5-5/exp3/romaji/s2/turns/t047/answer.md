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
kv={}
for h in [0,1,2,3,4,5]:
    q,k,v,o=headmats(0,h); A=q.T@k/8
    Uq,Sq,Vq=np.linalg.svd(A)
    key=Vq[0]*Sq[0]; kd=D.T@key
    kv[h]=kd
    print(h,'S1 %.3f'%Sq[0],'|key tok part| %.3f of %.3f'%(np.linalg.norm(kd),Sq[0]),'D coeffs',(kd/np.linalg.norm(kd)).round(2))
hs=[1,2,4,5]
print('cos between Pin heads key-token vectors')
for a in hs: print(a,[round(float(kv[a]@kv[b]/np.linalg.norm(kv[a])/np.linalg.norm(kv[b])),2) for b in hs])
# L0H5 OV: which token directions does it copy most
q,k,v,o=headmats(0,5); print('L0H5 OV gain on D',np.linalg.norm(o@v@D,axis=0).round(2))
```