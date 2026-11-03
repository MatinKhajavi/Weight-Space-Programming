```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def raw(l,j):
    s=slice(64*j,64*j+64); return W[f'W_q.{l}'][s],W[f'W_k.{l}'][s],W[f'W_v.{l}'][s],W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=raw(0,2)
A2=Wq.T@Wk
u,s,vt=np.linalg.svd(A2)
P=np.linalg.qr(np.concatenate([u[:,:16],vt[:16].T],1))[0]
P=np.linalg.svd(P)[0][:,:np.sum(np.linalg.svd(P,compute_uv=False)>0.5)]
uv,sv,vvt=np.linalg.svd(Wv); T=vvt[:20].T
# orthogonalize T against P
T=T-P@(P.T@T); T=np.linalg.svd(T,full_matrices=False)[0]
Q,_=np.linalg.qr(np.concatenate([P,T],1))
R=np.eye(384)-Q@Q.T
print('dims P',P.shape[1],'T',T.shape[1])
b=W['ln1_b.0']
print('ln1_b.0 energy in P,T,rest: %.3f %.3f %.3f'%(np.linalg.norm(P.T@b)**2,np.linalg.norm(T.T@b)**2,b@R@b))
for j in range(6):
    Wq,Wk,Wv,Wo=raw(0,j); A=Wq.T@Wk; tot=np.linalg.norm(A)**2
    blocks=np.array([[np.linalg.norm(X.T@A@Y)**2 if X is not None else 0 for Y in [P,T]] for X in [P,T]])/tot
    restq=np.linalg.norm(R@A)**2/tot; restk=np.linalg.norm(A@R)**2/tot
    OV=Wo@Wv; to=np.linalg.norm(OV)**2
    print(f'H{j} QK blocks (q rows P,T ; k cols P,T):',blocks.round(3).tolist(),' q-rest %.2f k-rest %.2f'%(restq,restk),
          '| OV input P %.2f T %.2f rest %.2f'%(np.linalg.norm(OV@P)**2/to,np.linalg.norm(OV@T)**2/to,np.linalg.norm(OV@R)**2/to))
```