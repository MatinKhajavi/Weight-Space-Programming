```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
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
# Positional basis from L0H3 QK (raw coords: these matrices already act on raw normalized residual)
q,k,v,o=headmats(0,3); U2,S2,V2=np.linalg.svd(q.T@k)
P3b=np.hstack([U2[:,:24],V2[:24].T])
q,k,v,o=headmats(0,0); U2,S2,V2=np.linalg.svd(q.T@k); P0b=np.hstack([U2[:,:4],V2[:4].T])
pins=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); pins.append(np.linalg.svd(o@v)[2][:3].T); U2,S2,V2=np.linalg.svd(q.T@k); pins.append(U2[:,:1])
Pall=np.linalg.svd(np.hstack([P3b,P0b]+pins),full_matrices=False)
print('pos basis sv',Pall[1][:50].round(2))
Pb=Pall[0][:,:40]
print('per-D energy in 40-dim pos union',(np.linalg.norm(Pb.T@D,axis=0)**2).round(2))
print('random expectation',40/384)
```