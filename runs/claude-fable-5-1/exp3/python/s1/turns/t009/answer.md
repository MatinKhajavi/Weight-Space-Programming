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
T=U[:,::-1][:,:22]
dirs={}
for j in range(6):
    q,k,v,o=head(0,j)
    for nm,Mx in [('q',q),('k',k),('v',v)]:
        u,s,vt=np.linalg.svd(Mx@T,full_matrices=False)
        print(f'L0H{j} {nm} on T sv: {s[:8]}')
        dirs[f'L0H{j}{nm}0']=vt[0]; dirs[f'L0H{j}{nm}1']=vt[1]
u,s,vt=np.linalg.svd(W['W_fc.0']@T,full_matrices=False)
print('W_fc.0 on T sv:',s)
for i in range(4): dirs[f'fc0_{i}']=vt[i]
u,s,vt=np.linalg.svd(W['W_fc.1']@T,full_matrices=False)
print('W_fc.1 on T sv:',s)
for i in range(3): dirs[f'fc1_{i}']=vt[i]
# induction heads' Q on T
for (l,j) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l,j)
    u,s,vt=np.linalg.svd(q@T,full_matrices=False); print(f'L{l}H{j} q on T sv {s[:8]}'); dirs[f'L{l}H{j}q0']=vt[0]
    u,s,vt=np.linalg.svd(v@T,full_matrices=False); print(f'L{l}H{j} v on T sv {s[:8]}'); dirs[f'L{l}H{j}v0']=vt[0]
keys=['L0H0k0','L0H0k1','L0H2k0','L0H2k1','L0H3k0','L0H3k1','L0H2q0','L0H3q0','L0H0v0','L0H2v0','L0H4v0','L0H4v1','fc0_0','fc0_1','fc0_2','fc0_3','fc1_0','fc1_1','L2H1q0','L3H1q0','L3H5q0','L2H1v0','L3H5v0']
print('abs cosines among selected T-directions:')
print('        '+' '.join(f'{k[:7]:>7}' for k in keys))
for a in keys:
    print(f'{a:>7} '+' '.join(f'{abs(dirs[a]@dirs[b]):7.2f}' for b in keys))
```