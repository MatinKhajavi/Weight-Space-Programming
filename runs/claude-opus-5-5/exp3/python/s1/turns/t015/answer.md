```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def top_eigs(M,k=12):
    e,v=np.linalg.eig(M)
    idx=np.argsort(-np.abs(e))[:k]
    return e[idx],v[:,idx]
E={}
for j in [1,5,2,3]:
    Q,K,V,O=head(0,j)
    e,v=top_eigs(Q.T@K,12)
    E[j]=(e,v)
    print(j,[f'{abs(x):.2f}@{np.degrees(np.angle(x)):.0f}' for x in e])
# plane overlaps between H1 and H5 eigenvectors
def plane(v):
    B=np.hstack([v.real[:,None],v.imag[:,None]]) if np.abs(v.imag).sum()>1e-8 else v.real[:,None]
    q,_=np.linalg.qr(B); return q
for a,b in [(1,5),(1,2),(5,2),(1,3),(5,3)]:
    ea,va=E[a]; eb,vb=E[b]
    print('overlap',a,b)
    for i in range(10):
        pa=plane(va[:,i])
        row=[]
        for k in range(10):
            pb=plane(vb[:,k])
            row.append(np.linalg.norm(pa.T@pb)**2/min(pa.shape[1],pb.shape[1]))
        print(f'{np.degrees(np.angle(ea[i])):5.0f}', ' '.join(f'{x:.2f}' for x in row))
    print('cols angles', [f'{np.degrees(np.angle(x)):.0f}' for x in eb[:10]])
```