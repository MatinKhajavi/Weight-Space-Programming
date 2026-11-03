```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=W['ln1_w.0']; rng=np.random.default_rng(5)
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def colspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return U[:,:k]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
def orth(A):
    Q,_=np.linalg.qr(A); return Q
T3=rowspace(blk('W_v',0,3),22); Tres=orth(np.diag(1/w0)@T3)
C3=colspace(P@W['W_o.0'][:,192:256]@blk('W_v',0,3)@np.diag(w0),18)
Cm=colspace(P@W['W_proj.0'],40)
Cp=colspace(np.hstack([P@W['W_o.0'][:,a*hd:(a+1)*hd]@blk('W_v',0,a)@np.diag(w0) for a in [1,2,4,5]]),16)
subs={'T':Tres,'M0':Cm,'H3':C3,'Pos':Cp}
def asym(C):
    ev=np.linalg.eigvals(C); return ev.real.sum()/np.abs(ev).sum()
def symfrac(C):  # for non-square: fraction of energy in sym part of (B^T C B) only when square
    return np.linalg.norm(C+C.T)/2/np.linalg.norm(C)
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        QK=(blk('W_q',l,j)@wl).T@(blk('W_k',l,j)@wl); tot=np.linalg.norm(QK)**2
        out=[]
        for qn in ['T','M0','H3','Pos']:
            for kn in ['T','M0','H3','Pos']:
                B=subs[qn].T@QK@subs[kn]; e=np.linalg.norm(B)**2/tot
                s=f"{qn}x{kn}:{e:.3f}"
                if qn==kn: s+=f"(a{asym(B):+.2f})"
                out.append(s)
        print(f"L{l}H{j} "+" ".join(out))
# null for diagonal blocks asym
for (l,j) in [(1,0),(2,0),(3,3)]:
    wl=np.diag(W[f'ln1_w.{l}']); QK=(blk('W_q',l,j)@wl).T@(blk('W_k',l,j)@wl)
    for nm in ['T','M0','H3','Pos']:
        a=[]
        for _ in range(20):
            R,_=np.linalg.qr(rng.standard_normal((d,d))); a.append(asym(subs[nm].T@R@QK@R.T@subs[nm]))
        print(f"null L{l}H{j} {nm}: {np.mean(a):+.2f}±{np.std(a):.2f}", end="  ")
    print()
```