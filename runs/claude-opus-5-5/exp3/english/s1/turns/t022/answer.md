```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
V2=W['W_v.0'][s2]*w0
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
U,_,_=np.linalg.svd(W['W_o.0'][:,s2]@V2); O2=U[:,:18]
def colsp(M,k): U,_,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
subs={'T':T,'O2':O2,'M0':colsp(W['W_proj.0'],40)}
for l,h in [(1,3),(1,4),(2,3),(2,2)]:
    s=slice(64*h,64*h+64)
    subs[f'L{l}H{h}']=colsp(W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*W[f'ln1_w.{l}']),16)
w=W['ln1_w.3']; h=1; s=slice(64,128)
B=(W['W_q.3'][s]*w).T@(W['W_k.3'][s]*w)
print('|B|',np.linalg.norm(B))
names=list(subs)
print('rows=query, cols=key; entries: spectral norm of block')
print('        '+' '.join(f'{n:>7}' for n in names))
for a in names:
    print(f'{a:7s} '+' '.join(f'{np.linalg.norm(subs[a].T@B@subs[c],2):7.2f}' for c in names))
# random baseline for 16-dim subspaces
rng=np.random.default_rng(0)
R1=np.linalg.qr(rng.normal(size=(384,18)))[0]; R2=np.linalg.qr(rng.normal(size=(384,18)))[0]
print('random 18x18 block',np.linalg.norm(R1.T@B@R2,2))
M=O2.T@B@O2; print('O2xO2 eig',np.round(np.sort(np.linalg.eigvals(M).real)[::-1][:10],2))
M=T.T@B@O2; print('T x O2 svals',np.round(np.linalg.svd(M,compute_uv=False)[:10],2))
```