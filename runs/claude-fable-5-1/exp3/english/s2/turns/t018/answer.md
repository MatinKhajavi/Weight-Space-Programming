```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from scipy.linalg import polar
d=384; hd=64; H=6
g=W["ln1_w.0"]
s2=slice(128,192)
A_full=(W["W_q.0"][s2]*g).T@(W["W_k.0"][s2]*g)
U,S,Vt=np.linalg.svd(A_full)
for r in [24,40]:
    Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:r],Vt[:r].T],axis=1)); P=Uc[:,:r]
    A=P.T@A_full@P
    Up,Hp=polar(A)
    ev=np.linalg.eigvals(Up); ang=np.sort(np.abs(np.angle(ev)))
    print(f"r={r}: rotation angles (deg) of polar part:", np.round(np.degrees(ang[::2]),1))
    ks=list(range(0,9))
    basis=[np.linalg.matrix_power(Up,k) for k in ks]
    # also include transposes (negative k)
    ksn=list(range(1,4)); basis_n=[np.linalg.matrix_power(Up.T,k) for k in ksn]
    Bm=np.array([b.ravel() for b in basis+basis_n]).T
    def fit(Bmat):
        coef,res,_,_=np.linalg.lstsq(Bm,Bmat.ravel(),rcond=None)
        r2=1-np.sum((Bm@coef-Bmat.ravel())**2)/np.sum(Bmat.ravel()**2)
        return coef,r2
    print("  basis: U^0..U^8, then U^-1..U^-3")
    for l in range(4):
        gl=W[f"ln1_w.{l}"]
        for j in range(H):
            s=slice(j*hd,(j+1)*hd); QK=(W[f"W_q.{l}"][s]*gl).T@(W[f"W_k.{l}"][s]*gl)
            B=P.T@QK@P
            coef,r2=fit(B)
            print(f"  L{l}H{j}: R2 {r2:.2f} coef", np.round(coef,2), " |B| %.2f"%np.linalg.norm(B))
```