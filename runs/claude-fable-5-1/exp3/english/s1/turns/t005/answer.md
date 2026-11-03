```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0=W["ln1_w.0"]
def eff(nm,j): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*w0
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T  # 384 x r orthonormal
def energy(A,S):  # fraction of row-space energy (Frobenius) of A inside subspace S
    return np.linalg.norm(A@S)**2/np.linalg.norm(A)**2
Q2=eff("W_q.0",2); K2=eff("W_k.0",2); V2=eff("W_v.0",2)
P = rsv(np.vstack([Q2,K2]),24)
T = rsv(V2,30)
print("principal angles cos between P(QK2,24) and T(V2,30):", np.linalg.svd(P.T@T,compute_uv=False)[:10])
print("energy of Q2 in P %.2f, K2 in P %.2f, V2 in P %.2f, V2 in T %.2f, Q2 in T %.2f K2 in T %.2f" % (energy(Q2,P),energy(K2,P),energy(V2,P),energy(V2,T),energy(Q2,T),energy(K2,T)))
# baseline energies for random 24- and 30-dim subspaces: 24/384=0.0625, 30/384=0.078
for j in range(6):
    for nm in ["W_q.0","W_k.0","W_v.0"]:
        A=eff(nm,j)
        print(f"L0H{j} {nm}: energy in P={energy(A,P):.2f}  in T={energy(A,T):.2f}")
# OV reading directions of broad heads: top right singular vectors of OV eff
for j in [0,1,3,4,5]:
    s=slice(64*j,64*(j+1))
    OVe = W["W_o.0"][:,s]@eff("W_v.0",j)
    u,sv,vt = np.linalg.svd(OVe)
    for r in range(3):
        v=vt[r]
        print(f"L0H{j} OV read dir {r} (sv {sv[r]:.2f}): |proj P|^2={np.linalg.norm(P.T@v)**2:.2f} |proj T|^2={np.linalg.norm(T.T@v)**2:.2f}")
# MLP0 reading
F=W["W_fc.0"]*W["ln2_w.0"]
print("W_fc.0 energy in P %.2f in T %.2f" % (energy(F,P),energy(F,T)))
# How do P and T relate to the ln1_w.0 'big' dims?
big = w0>0.3
print("P energy on big dims: %.2f (big dims fraction %.2f)" % ((P[big]**2).sum()/ (P**2).sum(), big.mean()))
print("T energy on big dims: %.2f" % ((T[big]**2).sum()/ (T**2).sum()))
```