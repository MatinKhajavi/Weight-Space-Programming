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
    M=(q.T@k)@OV[(0,5)]; S=(M+M.T)/2; Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
T=U[:,::-1][:,:22]; PT=T@T.T
# define A and B directions (in full residual coords)
q0,k0,v0,o0=head(0,0); u,s,vt=np.linalg.svd(k0@T,full_matrices=False); A=T@vt[0]
q3,k3,v3,o3=head(0,3); u,s,vt=np.linalg.svd(k3@T,full_matrices=False); B=T@vt[0]
B=B-A*(A@B); B/=np.linalg.norm(B)
print('A.B',A@B)
# sign conventions: orient A, B so that MLP0 reading is ... arbitrary; keep as is
for l in range(4):
    for j in range(6):
        q,k,v,o=head(l,j)
        QK=q.T@k
        aa=A@QK@A; ab=A@QK@B; ba=B@QK@A; bb=B@QK@B
        # rest of T
        R=T - np.outer(A,A@T) - np.outer(B,B@T)
        uR,sR,vR=np.linalg.svd(R,full_matrices=False); R=uR[:,:20]
        trR=np.trace(R.T@QK@R); nR=np.linalg.norm(R.T@QK@R)
        # how much A,B in q and k rows
        qa=np.linalg.norm(q@A)**2/np.linalg.norm(q)**2; qb=np.linalg.norm(q@B)**2/np.linalg.norm(q)**2
        ka=np.linalg.norm(k@A)**2/np.linalg.norm(k)**2; kb=np.linalg.norm(k@B)**2/np.linalg.norm(k)**2
        va=np.linalg.norm(v@A)**2/np.linalg.norm(v)**2; vb=np.linalg.norm(v@B)**2/np.linalg.norm(v)**2
        # A/B -> position cross terms: |A QK P| etc
        print(f'L{l}H{j}: QK[AA]={aa:6.2f} [AB]={ab:6.2f} [BA]={ba:6.2f} [BB]={bb:6.2f} | restT tr={trR:6.2f} |.|={nR:5.2f} | q:A {qa:.3f} B {qb:.3f}  k:A {ka:.3f} B {kb:.3f}  v:A {va:.3f} B {vb:.3f}')
```