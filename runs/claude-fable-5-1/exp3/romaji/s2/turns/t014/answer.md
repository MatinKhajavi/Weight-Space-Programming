```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=W['ln1_w.0']
rng=np.random.default_rng(3)
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
def orth(A):
    Q,_=np.linalg.qr(A); return Q
T3=rowspace(blk('W_v',0,3),22); Tres=orth(np.diag(1/w0)@T3)
OV3=W['W_o.0'][:,192:256]@blk('W_v',0,3)@np.diag(w0)
def asym(C):
    ev=np.linalg.eigvals(C); return ev.real.sum()/np.abs(ev).sum()
# null: random rotation of QK of each head
for (l,j) in [(1,0),(1,1),(3,3),(2,0)]:
    wl=np.diag(W[f'ln1_w.{l}']); QK=(blk('W_q',l,j)@wl).T@(blk('W_k',l,j)@wl)
    nd=[]; ni=[]
    for _ in range(30):
        R,_=np.linalg.qr(rng.standard_normal((d,d))); QKr=R@QK@R.T
        nd.append(asym(Tres.T@QKr@Tres)); ni.append(asym(Tres.T@QKr@P@OV3@Tres))
    print(f"L{l}H{j} null direct asym mean±sd {np.mean(nd):.2f}±{np.std(nd):.2f}; null ind asym {np.mean(ni):.2f}±{np.std(ni):.2f}")
# magnitude: fraction of QK energy (Frobenius) within T-T block, and within (T, OV3 T) block
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        QK=(blk('W_q',l,j)@wl).T@(blk('W_k',l,j)@wl)
        tt=np.linalg.norm(Tres.T@QK@Tres)**2/np.linalg.norm(QK)**2
        # key subspace of OV3 output
        K3=orth(P@OV3@Tres)
        ti=np.linalg.norm(Tres.T@QK@K3)**2/np.linalg.norm(QK)**2
        # position-ish key subspace: outputs of L0H1,2,4,5
        srcs=np.hstack([W['W_o.0'][:,a*hd:(a+1)*hd]@blk('W_v',0,a)@np.diag(w0) for a in [1,2,4,5]])
        Kp=rowspace(srcs.T,40)
        qp=np.linalg.norm(QK@Kp)**2/np.linalg.norm(QK)**2
        print(f"L{l}H{j}: frac QK energy T-T={tt:.3f} (rand {(22/384)**2:.4f}); T-(OV3 T)={ti:.3f}; keys in L0H1/2/4/5-out(40)={qp:.3f} (rand {40/384:.3f})")
```