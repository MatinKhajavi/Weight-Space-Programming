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
T=U[:,::-1][:,:22]; PT=T@T.T; PP=np.eye(d)-PT
rows=np.vstack([head(0,1)[0],head(0,1)[1],head(0,5)[0],head(0,5)[1]])@PP
u,s,vt=np.linalg.svd(rows,full_matrices=False); P0=vt[:40].T
q0,k0,v0,o0=head(0,0); u,s,vt=np.linalg.svd(k0@T,full_matrices=False); A=T@vt[0]
q3,k3,v3,o3=head(0,3); u,s,vt=np.linalg.svd(k3@T,full_matrices=False); B=T@vt[0]
B=B-A*(A@B); B/=np.linalg.norm(B)
# orthonormal basis of T with A,B first
R=T-np.outer(A,A@T)-np.outer(B,B@T); uR,sR,vR=np.linalg.svd(R,full_matrices=False); Tb=np.column_stack([A,B,uR[:,:20]])
for l in range(2):
    for j in range(6):
        q,k,v,o=head(l,j); QK=q.T@k
        TT=Tb.T@QK@Tb; PTb=P0.T@QK@Tb; TPb=Tb.T@QK@P0
        sTT=np.linalg.svd(TT,compute_uv=False)
        # which content directions are attended given positional queries: column norms of P0^T QK Tb (per content dir)
        colK=np.linalg.norm(PTb,axis=0)   # for each content dir on K side
        rowQ=np.linalg.norm(TPb,axis=1)   # content on Q side vs positional keys
        print(f'L{l}H{j}: TT sv {sTT[:5]} tr {np.trace(TT):.2f} | pos-query x content-key norms: A {colK[0]:.2f} B {colK[1]:.2f} others mean {colK[2:].mean():.2f} max {colK[2:].max():.2f} | content-query x pos-key: A {rowQ[0]:.2f} B {rowQ[1]:.2f} others mean {rowQ[2:].mean():.2f} max {rowQ[2:].max():.2f}')
    # also: does the head's V read A/B/others
for l in range(2):
    for j in range(6):
        q,k,v,o=head(l,j)
        cn=np.linalg.norm(v@Tb,axis=0)
        print(f'L{l}H{j} V content gains: A {cn[0]:.2f} B {cn[1]:.2f} others mean {cn[2:].mean():.2f} min {cn[2:].min():.2f} max {cn[2:].max():.2f}')
```