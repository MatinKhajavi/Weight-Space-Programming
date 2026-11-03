```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
V2=W['W_v.0'][s2]*w0
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
U,_,_=np.linalg.svd(W['W_o.0'][:,s2]@V2); O2=U[:,:18]
UF=[]
for h in [0,1,4,5]:
    s=slice(64*h,64*h+64); U_,_,_=np.linalg.svd(W['W_o.0'][:,s]@(W['W_v.0'][s]*w0)); UF.append(U_[:,:3])
U3,_,_=np.linalg.svd(np.hstack(UF)); MD=U3[:,:4]
def tr_ratio(M): 
    ev=np.linalg.eigvals(M); return np.sum(ev.real)/np.sum(np.abs(ev))
for l in [1,2,3]:
    w=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        OV=W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*w)
        # which subspaces does it write into
        tot=np.sum(OV**2)
        wr={k:np.sum((S.T@OV)**2)/tot for k,S in [('T',T),('O2',O2),('MD',MD)]}
        print(f'L{l}H{h} OV O2->O2 pos={tr_ratio(O2.T@OV@O2):+.2f} |{np.linalg.norm(O2.T@OV@O2):.2f}|  T->T pos={tr_ratio(T.T@OV@T):+.2f} |{np.linalg.norm(T.T@OV@T):.2f}|  O2->T |{np.linalg.norm(T.T@OV@O2):.2f}| T->O2 |{np.linalg.norm(O2.T@OV@T):.2f}|  write fracs', {k:round(float(v),2) for k,v in wr.items()}, f'|OV|={np.sqrt(tot):.1f}')
```