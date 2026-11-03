```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
Vt=np.vstack([head(0,j)[2]/np.linalg.norm(head(0,j)[2]) for j in [1,5]])
_,sv,vt=np.linalg.svd(Vt,full_matrices=False)
Ptok=vt[:26].T
# positional basis: top dirs read by Q/K of heads 1,5 that are NOT in Ptok
Pp=np.hstack([np.linalg.svd(head(0,j)[i],full_matrices=False)[2][:8].T for j in [1,5] for i in [0,1]])
Pp=Pp-Ptok@(Ptok.T@Pp); Ppos=np.linalg.svd(Pp,full_matrices=False)[0][:,:16]
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    Q,K,V,O=head(0,j)
    def en(M,P): return (np.linalg.norm(M@P)**2/np.linalg.norm(M)**2).round(2)
    QK=Q.T@K
    tt=np.linalg.norm(Ptok.T@QK@Ptok); tp=np.linalg.norm(Ptok.T@QK@Ppos); pt=np.linalg.norm(Ppos.T@QK@Ptok); pp=np.linalg.norm(Ppos.T@QK@Ppos)
    q0=W['W_q.0'][s]@W['ln1_b.0']
    print(f'L0H{j} Q tok/pos {en(Q,Ptok)}/{en(Q,Ppos)}  K tok/pos {en(K,Ptok)}/{en(K,Ppos)}  V tok/pos {en(V,Ptok)}/{en(V,Ppos)} | QK blocks qtok-ktok {tt:.2f} qtok-kpos {tp:.2f} qpos-ktok {pt:.2f} qpos-kpos {pp:.2f}')
    Mk=QK@Ptok
    print('    key-token sv', np.linalg.svd(Mk,compute_uv=False)[:6].round(2), ' bias pref norm %.2f'%np.linalg.norm(q0@K@Ptok/8))
```