```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)])
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1); Fa=F[nf>1.0]
Fr=Fa-(Fa@Hout)@Hout.T
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:26].T   # directions in residual (raw) space
q,k,v,o=headmats(0,3)
U2,S2,Vt2=np.linalg.svd(q.T@k)
for npos in [4,8,12,16,20,24]:
    Posraw=np.linalg.qr(np.hstack([U2[:,:npos],Vt2[:npos].T])/lw0[:,None])[0][:,:npos]
    c=np.linalg.svd(D.T@Posraw,compute_uv=False)
    print(npos,'cos of D(26) with Pos',c.round(2))
# per direction: fraction in Pos(16)
Posraw=np.linalg.qr(np.hstack([U2[:,:16],Vt2[:16].T])/lw0[:,None])[0][:,:16]
print('per-dir pos frac',(np.linalg.norm(Posraw.T@D,axis=0)**2).round(2))
# heads 1,2,4,5 V input (Pin) raw
ins=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); ins.append(np.linalg.svd(o@v)[2][:2].T)
Pin=np.linalg.qr(np.linalg.svd(np.hstack(ins))[0][:,:2]/lw0[:,None])[0]
print('per-dir Pin frac',(np.linalg.norm(Pin.T@D,axis=0)**2).round(2))
# L0H3 OV input (token) raw
U3,S3,V3=np.linalg.svd(o@v)
q,k,v,o=headmats(0,3); U3,S3,V3=np.linalg.svd(o@v); Tokraw=np.linalg.qr(V3[:20].T/lw0[:,None])[0]
print('per-dir Tok frac',(np.linalg.norm(Tokraw.T@D,axis=0)**2).round(2))
print('sv',S[:30].round(2))
```