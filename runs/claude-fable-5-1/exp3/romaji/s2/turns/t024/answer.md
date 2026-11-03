```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=np.diag(W['ln1_w.0'])
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
OV5=P@W['W_o.0'][:,5*hd:6*hd]@blk('W_v',0,5)@w0
U,S,Vt=np.linalg.svd(OV5)
for r in range(3):
    u=U[:,r]; top=np.argsort(-np.abs(u))[:10]
    print(f"L0H5 out dir{r} sv={S[r]:.1f}: top dims {top} loads {u[top]}")
b1=W['ln1_b.1']; b2=W['ln1_b.2']; w1=W['ln1_w.1']
print("b1 projected on C5 dirs:", [round(float(U[:,r]@b1),2) for r in range(4)], " |b1|=",np.linalg.norm(b1).round(2))
print("b2 projected on C5 dirs:", [round(float(U[:,r]@b2),2) for r in range(4)], " |b2|=",np.linalg.norm(b2).round(2))
print("w1 on top dims of dir0:", w1[np.argsort(-np.abs(U[:,0]))[:6]], "w1 median", np.median(w1))
# W_o.0 row norms for dims 39,58,99,21 by head
for dim in [39,58,99,21,184,383]:
    print(dim, "W_o.0 row norm by head:", [round(float(np.linalg.norm(W['W_o.0'][dim,j*hd:(j+1)*hd])),2) for j in range(6)], " proj0 row norm", round(float(np.linalg.norm(W['W_proj.0'][dim])),2), " w0", round(float(W['ln1_w.0'][dim]),2))
# key-side: does W_k.1 b1 (constant query term) align with the W_k.1 read of C5?  Compute per head: cos(W_q b1-direction effect) 
for j in range(6):
    wl=np.diag(w1); q=blk('W_q',1,j)@wl; k=blk('W_k',1,j)@wl
    # constant query vector from bias: q_b = W_q.1 @ b1 (b1 added after w scaling, so no wl)
    qb=blk('W_q',1,j)@b1
    # key direction preferred by constant query: W_k^T qb ; compare with C5 dirs
    kd=k.T@qb; 
    print(f"L1H{j}: |qb|={np.linalg.norm(qb):.2f}, |W_q wl| rows avg {np.linalg.norm(q,axis=1).mean():.2f}; fraction of (W_k^T qb) in C5(4): {np.linalg.norm(U[:,:4].T@kd)**2/np.linalg.norm(kd)**2:.2f}; loads on dir0,1: {U[:,0]@kd:+.2f} {U[:,1]@kd:+.2f}")
```