```python
import numpy as np
w0=W['ln1_w.0']
def rows(nm,l,h): 
    s=slice(64*h,64*h+64); return W[f'{nm}.{l}'][s]
s2=slice(128,192)
Q2=W['W_q.0'][s2]*w0; K2=W['W_k.0'][s2]*w0; V2=W['W_v.0'][s2]*w0
_,_,Vt=np.linalg.svd(np.vstack([Q2,K2]),full_matrices=False); P=Vt[:22].T
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
OV2=W['W_o.0'][:,s2]@V2
U,_,_=np.linalg.svd(OV2); O2=U[:,:18]
UF=[]
for h in [0,1,4,5]:
    s=slice(64*h,64*h+64); U,_,_=np.linalg.svd(W['W_o.0'][:,s]@(W['W_v.0'][s]*w0)); UF.append(U[:,:3])
MD,_=np.linalg.qr(np.hstack(UF)); 
_,sMD,_=np.linalg.svd(np.hstack(UF)); 
U3,_,_=np.linalg.svd(np.hstack(UF)); MD=U3[:,:4]
s3=slice(192,256)
OV3=W['W_o.0'][:,s3]@(W['W_v.0'][s3]*w0); U,_,_=np.linalg.svd(OV3); O3=U[:,:5]
subs={'P':P,'T':T,'O2':O2,'MD':MD,'O3':O3}
print('overlaps P-T',np.round(np.linalg.svd(P.T@T,compute_uv=False)[:5],2),' T-O2',np.round(np.linalg.svd(T.T@O2,compute_uv=False)[:5],2), ' P-MD', np.round(np.linalg.svd(P.T@MD,compute_uv=False),2))
for l in [1,2,3]:
    w=W[f'ln1_w.{l}']
    for h in range(6):
        line=f'L{l}H{h}'
        for nm in ['W_q','W_k','W_v']:
            M=rows(nm,l,h)*w; tot=np.sum(M**2)
            line+=f' | {nm[-1]}: '+' '.join(f'{k}={np.sum((M@S)**2)/tot:.2f}' for k,S in subs.items())
        print(line)
```