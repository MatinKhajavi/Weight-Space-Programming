```python
import numpy as np
U3,S3,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); OUT=U3[:,:23]
w0=W['ln1_w.0']; s2=slice(128,192)
V2=W['W_v.0'][s2]*w0
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
_,_,Vt=np.linalg.svd(np.vstack([W['W_q.0'][s2]*w0,W['W_k.0'][s2]*w0]),full_matrices=False); P=Vt[:22].T
U,_,_=np.linalg.svd(W['W_o.0'][:,s2]@V2); O2=U[:,:18]
def colsp(M,k): U,_,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
def rowsp(M,k): _,_,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
subs={'T':T,'P':P,'O2':O2,'OUT23':OUT}
for l in range(4):
    s=slice(64*l*0,0)
for l in [0,1,2]:
    for h in range(6):
        s=slice(64*h,64*h+64)
        OV=W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*W[f'ln1_w.{l}'])
        subs[f'L{l}H{h}']=colsp(OV,16)
    subs[f'M{l}']=colsp(W[f'W_proj.{l}'],64)
w=W['ln1_w.3']
for h in range(6):
    s=slice(64*h,64*h+64)
    Q=W['W_q.3'][s]*w; K=W['W_k.3'][s]*w; V=W['W_v.3'][s]*w
    line=f'L3H{h}'
    for nm,M in [('q',Q),('k',K),('v',V)]:
        tot=np.sum(M**2)
        fr={k:np.sum((M@S)**2)/tot/(S.shape[1]/384) for k,S in subs.items()}
        top=sorted(fr.items(),key=lambda t:-t[1])[:6]
        line+=f' | {nm}: '+' '.join(f'{k}={v:.1f}x' for k,v in top)
    print(line)
```