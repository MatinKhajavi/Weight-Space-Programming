```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
Q3=hm(0,'q',3)*ln0; K3=hm(0,'k',3)*ln0
_,_,VtP=np.linalg.svd(np.vstack([Q3,K3])); P=VtP[:24].T
V3=hm(0,'v',3)*ln0
_,_,VtT=np.linalg.svd(V3); T=VtT[:24].T
# Are P and T orthogonal-ish?
print("overlap P,T:", np.linalg.norm(P.T@T)**2/24)
for j in [1,2,4,5,0]:
    Q=hm(0,'q',j)*ln0; K=hm(0,'k',j)*ln0
    M=Q.T@K
    U,S,Vt=np.linalg.svd(M)
    print(f"L0H{j}: S={S[:4]}")
    for c in range(3):
        u=U[:,c]; v=Vt[c]
        print(f"   comp{c}: q-dir in P={np.linalg.norm(P.T@u)**2:.2f} in T={np.linalg.norm(T.T@u)**2:.2f} | k-dir in P={np.linalg.norm(P.T@v)**2:.2f} in T={np.linalg.norm(T.T@v)**2:.2f}")
    V=hm(0,'v',j)*ln0
    OV=W['W_o.0'][:,j*64:(j+1)*64]@V
    Uo,So,Vto=np.linalg.svd(OV)
    for c in range(3):
        v=Vto[c]
        print(f"   OV in-dir{c} (sv {So[c]:.2f}): in P={np.linalg.norm(P.T@v)**2:.2f} in T={np.linalg.norm(T.T@v)**2:.2f}")
# k-dirs of H1,H2,H4 in T: which T-direction? compare among them
ks=[]
for j in [1,2,4]:
    Q=hm(0,'q',j)*ln0; K=hm(0,'k',j)*ln0
    U,S,Vt=np.linalg.svd(Q.T@K); ks.append(Vt[0]*np.sign(U[0,0]))
print("k-dir cos H1-H2 H1-H4 H2-H4:", ks[0]@ks[1], ks[0]@ks[2], ks[1]@ks[2])
# Compare the k-dir with directions the MLP0 reads most / with the 'mean' direction (all ones normalized)
ones=np.ones(384)/np.sqrt(384)
print("k-dir cos with all-ones:", [float(k@ones) for k in ks])
```