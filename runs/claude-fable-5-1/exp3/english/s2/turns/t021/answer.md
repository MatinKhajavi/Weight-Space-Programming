```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from scipy.linalg import polar
d=384; hd=64; H=6
g=W["ln1_w.0"]
s2=slice(128,192)
A_full=(W["W_q.0"][s2]*g).T@(W["W_k.0"][s2]*g)
U,S,Vt=np.linalg.svd(A_full)
r=30
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:r],Vt[:r].T],axis=1)); P=Uc[:,:r]
A=P.T@A_full@P
Up,Hp=polar(A)
ev,evec=np.linalg.eig(Up)
ang=np.degrees(np.abs(np.angle(ev)))
order=np.argsort(ang); ev=ev[order]; evec=evec[:,order]; ang=ang[order]
print("angles:", np.round(ang,1))
# the polar 'gain' H along eigvecs: how strongly each frequency is used by L0H2
gain=np.real(np.einsum('ij,ji->i',evec.conj().T,Hp@evec))
print("L0H2 gain per eigvec:", np.round(gain,1))
def spectrum(u):
    c=np.linalg.solve(evec,u)  # coordinates (evec not orthogonal in general; but Up normal -> unitary eigvecs)
    e=np.abs(c)**2; e=e/e.sum()
    # merge into angle bins
    bins=[0,5,20,45,90,135,181]
    return [round(float(e[(ang>=a)&(ang<b)].sum()),2) for a,b in zip(bins[:-1],bins[1:])]
print("angle bins [0-5,5-20,20-45,45-90,90-135,135-180]")
# L0H3 top direction and broad heads' top directions
for j in [0,1,3,4,5]:
    s=slice(j*hd,(j+1)*hd); QK=(W["W_q.0"][s]*g).T@(W["W_k.0"][s]*g); B=P.T@QK@P
    u,sv,vt=np.linalg.svd(B)
    print(f"L0H{j}: top sv {sv[:3]}; query-dir spectrum {spectrum(u[:,0])} key-dir spectrum {spectrum(vt[0])}")
# layer1-3: constant-query key direction and top P-block singular directions
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]; bl=W[f"ln1_b.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd); Wq=W[f"W_q.{l}"][s]*gl; Wk=W[f"W_k.{l}"][s]*gl
        kdir=P.T@((W[f"W_q.{l}"][s]@bl)@Wk)
        B=P.T@(Wq.T@Wk)@P; u,sv,vt=np.linalg.svd(B)
        print(f"L{l}H{j}: const-query key spectrum {spectrum(kdir/np.linalg.norm(kdir))} | top P-block sv {sv[:3]} key-dir spectrum {spectrum(vt[0])} query-dir {spectrum(u[:,0])}")
```