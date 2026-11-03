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
u,s,vt=np.linalg.svd(rows,full_matrices=False); P0=vt[:40].T; PP0=P0@P0.T
u,s,vt=np.linalg.svd(W['W_proj.0']); M0=u[:,:40]; PM0=M0@M0.T
outsub={}
for j in range(6):
    u,s,vt=np.linalg.svd(OV[(0,j)]); outsub[j]=u[:,:8]
def describe(vec):
    vec=vec/np.linalg.norm(vec)
    parts=[f'T:{np.linalg.norm(PT@vec)**2:.2f}', f'P0:{np.linalg.norm(PP0@vec)**2:.2f}', f'M0:{np.linalg.norm(PM0@vec)**2:.2f}']
    for j in range(6): parts.append(f'O{j}:{np.linalg.norm(outsub[j].T@vec)**2:.2f}')
    return ' '.join(parts)
for (l2,h2),(l1,h1) in [((1,1),(0,4)),((1,3),(0,4)),((1,1),(0,0)),((1,3),(0,0)),((1,0),(0,4)),((2,2),(0,4)),((2,3),(0,0))]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(l1,h1)]
    u,s,vt=np.linalg.svd(M)
    print(f'L{l2}H{h2} <- L{l1}H{h1}: sv {s[:6]}')
    for i in range(3):
        print(f'   Qside{i}: {describe(u[:,i])}   | Kside(V-input){i}: {describe(vt[i])}')
# For L0H4 and L0H0: top OV input (V) and output directions description
for j in [0,4,5,1]:
    q,k,v,o=head(0,j)
    uu,ss,vv=np.linalg.svd(OV[(0,j)])
    print(f'L0H{j} OV sv {ss[:6]}')
    for i in range(3):
        print(f'   in{i}: {describe(vv[i])}')
    # QK svd
    uu,ss,vv=np.linalg.svd(q.T@k)
    print(f'L0H{j} QK sv {ss[:6]}')
    for i in range(3):
        print(f'   Q{i}: {describe(uu[:,i])} | K{i}: {describe(vv[i])}')
```