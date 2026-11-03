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
u,s,vt=np.linalg.svd(rows,full_matrices=False); P0=vt[:48].T
A={}
for j in range(6):
    q,k,v,o=head(0,j); A[j]=P0.T@(q.T@k)@P0
# eigen-decomposition of A5
lam,V=np.linalg.eig(A[5])
order=np.argsort(-abs(lam)); lam=lam[order]; V=V[:,order]
print('L0H5 eigenvalues (|λ|, angle deg):')
print(np.round(np.abs(lam[:24]),2)); print(np.round(np.degrees(np.angle(lam[:24])),0))
# planes from complex eigenvectors; compute A1,A0,A2,A3 restricted to each plane: 2x2 -> angle and magnitude
used=set()
print('plane: |λ5| θ5 | for A1: mag θ1 | A0 | A2 | A3  (θ = atan2(antisym, sym) of 2x2 restriction)')
for i in range(len(lam)):
    if i in used or abs(lam[i])<1.0: continue
    if abs(lam[i].imag)<1e-6: continue
    vec=V[:,i]; re=vec.real; im=vec.imag
    Bp=np.linalg.qr(np.stack([re,im],1))[0]
    used.add(i)
    # find conjugate partner
    for jx in range(len(lam)):
        if jx!=i and abs(lam[jx]-np.conj(lam[i]))<1e-6: used.add(jx)
    outs=[]
    for hidx in [5,1,0,2,3]:
        R=Bp.T@A[hidx]@Bp
        sym=(R[0,0]+R[1,1])/2; asym=(R[0,1]-R[1,0])/2
        outs.append(f'{np.hypot(sym,asym):5.2f} {np.degrees(np.arctan2(asym,sym)):6.0f}')
    print(f'|λ5|={abs(lam[i]):5.2f} θ5={np.degrees(np.angle(lam[i])):6.0f} || '+' | '.join(outs))
```