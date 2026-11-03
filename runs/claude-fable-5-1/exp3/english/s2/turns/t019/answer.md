```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
U,S,Vt=np.linalg.svd((W["W_q.0"][s2]*g).T@(W["W_k.0"][s2]*g))
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
def OV(l,j):
    s=slice(j*hd,(j+1)*hd); gl=W[f"ln1_w.{l}"]; return W[f"W_o.{l}"][:,s]@(W[f"W_v.{l}"][s]*gl)
def topcols(M,k): u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
Ob=topcols(np.concatenate([OV(0,j) for j in [0,1,4,5]],axis=1),30)
R=Ob-P@(P.T@Ob); Obq,_=np.linalg.qr(R)
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]; bl=W[f"ln1_b.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd); Wq=W[f"W_q.{l}"][s]*gl; Wk=W[f"W_k.{l}"][s]*gl; QK=Wq.T@Wk
        B=P.T@QK@P; ev=np.linalg.eigvalsh((B+B.T)/2)
        Bo=Obq.T@QK@Obq; evo=np.linalg.eigvalsh((Bo+Bo.T)/2)
        qc=W[f"W_q.{l}"][s]@bl  # constant query from LN bias
        kc=W[f"W_k.{l}"][s]@bl
        # constant query's key-direction
        kdir=qc@Wk   # (384,) : score contribution per unit normalized residual at key
        print(f"L{l}H{j}: P-P sym eig top {ev[-3:][::-1]} bot {ev[:3]} asym/sym {np.linalg.norm(B-B.T)/np.linalg.norm(B+B.T):.2f} | Ob-Ob sym top {evo[-2:][::-1]} bot {evo[:2]} | |q_const| {np.linalg.norm(qc):.2f} |k_const| {np.linalg.norm(kc):.2f} ; q_const key-dir energy in P {np.linalg.norm(P.T@kdir)**2/np.linalg.norm(kdir)**2:.2f} Ob {np.linalg.norm(Obq.T@kdir)**2/np.linalg.norm(kdir)**2:.2f} |kdir| {np.linalg.norm(kdir):.2f}")
```