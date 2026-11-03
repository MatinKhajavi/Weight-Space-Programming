```python
import numpy as np
w0=W['ln1_w.0']
s2=slice(128,192)
Q2=W['W_q.0'][s2]*w0; K2=W['W_k.0'][s2]*w0; V2=W['W_v.0'][s2]*w0
_,_,Vt=np.linalg.svd(np.vstack([Q2,K2]),full_matrices=False); P=Vt[:22].T
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T0=Vt[:18].T
U,_,_=np.linalg.svd(W['W_o.0'][:,s2]@V2); O20=U[:,:18]
UF=[]
for h in [0,1,4,5]:
    s=slice(64*h,64*h+64); U,_,_=np.linalg.svd(W['W_o.0'][:,s]@(W['W_v.0'][s]*w0)); UF.append(U[:,:3])
U3,_,_=np.linalg.svd(np.hstack(UF)); MD0=U3[:,:4]
def orth_against(A,Bs):
    for B in Bs: A=A-B@(B.T@A)
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return U[:,S>0.3*S[0]], S
MD,_=orth_against(MD0,[])
P_,sp=orth_against(P,[MD]); 
T_,st=orth_against(T0,[MD,P_]); 
O2_,so=orth_against(O20,[MD,P_,T_])
print('dims',MD.shape[1],P_.shape[1],T_.shape[1],O2_.shape[1])
subs={'MD':MD,'P':P_,'T':T_,'O2':O2_}
for l in [1,2]:
    w=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        B=(W[f'W_q.{l}'][s]*w).T@(W[f'W_k.{l}'][s]*w)
        tot=np.linalg.norm(B)
        out=[]
        for a,A in subs.items():
            out.append(' '.join(f'{np.linalg.norm(A.T@B@C)/tot:.2f}' for c,C in subs.items()))
        print(f'L{l}H{h} rows=q[MD,P,T,O2] cols=k[MD,P,T,O2]:  ' + ' || '.join(out))
```