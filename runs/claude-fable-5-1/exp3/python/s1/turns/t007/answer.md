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
# positional subspace proxy: from L0H1 and L0H5 Q,K rows, complement of T
rows=np.vstack([head(0,1)[0],head(0,1)[1],head(0,5)[0],head(0,5)[1]])@PP
u,s,vt=np.linalg.svd(rows,full_matrices=False)
print('sv of L0H1/L0H5 QK rows on complement of T:',s[:60])
P0=vt[:40].T   # 40-dim positional proxy
PP0=P0@P0.T
# what do the layer-0 Q/K of other heads read?
for j in range(6):
    q,k,v,o=head(0,j)
    print(f'L0H{j}: Q in T {np.linalg.norm(q@PT)**2/np.linalg.norm(q)**2:.2f} in P0 {np.linalg.norm(q@PP0)**2/np.linalg.norm(q)**2:.2f} | K in T {np.linalg.norm(k@PT)**2/np.linalg.norm(k)**2:.2f} in P0 {np.linalg.norm(k@PP0)**2/np.linalg.norm(k)**2:.2f} | V in T {np.linalg.norm(v@PT)**2/np.linalg.norm(v)**2:.2f} in P0 {np.linalg.norm(v@PP0)**2/np.linalg.norm(v)**2:.2f}')
# output subspaces of L0 heads (top-8 left sv of OV) and MLP0 (top-40 left sv of W_proj.0)
outsub={}
for j in range(6):
    u,s,vt=np.linalg.svd(OV[(0,j)]); outsub[f'O{j}']=u[:,:8]
u,s,vt=np.linalg.svd(W['W_proj.0']); outsub['M0']=u[:,:40]
print('W_proj.0 sv:',s[:50])
for l in range(1,4):
    for j in range(6):
        q,k,v,o=head(l,j)
        line=f'L{l}H{j} '
        for nm,Mx in [('Q',q),('K',k),('V',v)]:
            tot=np.linalg.norm(Mx)**2
            parts=[f'T:{np.linalg.norm(Mx@PT)**2/tot:.2f}',f'P0:{np.linalg.norm(Mx@PP0)**2/tot:.2f}']
            for key,B in outsub.items():
                parts.append(f'{key}:{np.linalg.norm(Mx@B)**2/tot:.2f}')
            line+=f'| {nm} '+' '.join(parts)+' '
        print(line)
```