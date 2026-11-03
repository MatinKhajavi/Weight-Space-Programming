```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
Ssum=np.zeros((d,d))
for (l2,h2) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(0,5)]
    S=(M+M.T)/2
    Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
T=U[:,::-1][:,:22]; PT=T@T.T; PP=np.eye(d)-PT
names=[f'L{l}H{j}' for l in range(4) for j in range(H)]
QKs=[]; QKp=[]
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); A=q.T@k; QKs.append(A); QKp.append(PP@A@PP)
def cosm(A,B): return (A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)
print('cosine between QK restricted to non-token subspace (rows: heads)')
C=np.array([[cosm(QKp[a],QKp[b]) for b in range(24)] for a in range(24)])
print('      '+' '.join(f'{n:>5}' for n in names))
for a in range(24):
    print(f'{names[a]:>5} '+' '.join(f'{C[a,b]:5.2f}' for b in range(24)))
print('cosine with transpose of L0H5 QK (non-token part):')
print([f'{names[a]}:{cosm(QKp[a],QKp[5].T):.2f}' for a in range(24)])
# OV singular spectra of L0H5, L0H1 restricted to T
for (l,j) in [(0,5),(0,1),(0,0),(0,4),(0,2),(0,3)]:
    q,k,v,o=head(l,j)
    print(names[l*6+j],'sv of V on T:',np.linalg.svd(v@T,compute_uv=False)[:24])
    print(names[l*6+j],'sv of V on P:',np.linalg.svd(v@PP,compute_uv=False)[:24])
```