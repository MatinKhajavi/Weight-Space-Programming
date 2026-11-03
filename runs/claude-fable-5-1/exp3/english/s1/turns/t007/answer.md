```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W["ln1_w.0"]
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
def lsv(A,r):
    u,_,_=np.linalg.svd(A,full_matrices=False); return u[:,:r]
def energy(A,S): return np.linalg.norm(A@S)**2/np.linalg.norm(A)**2
Q2=eff("W_q.0",2); K2=eff("W_k.0",2); V2=eff("W_v.0",2)
P = rsv(np.vstack([Q2,K2]),24); T = rsv(V2,30)
OVe = {j: W["W_o.0"][:,64*j:64*(j+1)]@eff("W_v.0",j) for j in range(6)}
Ub = np.hstack([lsv(OVe[j],3) for j in [0,1,4,5]])   # 12 dims output of broad heads
Ub,_ = np.linalg.qr(Ub)
U2 = lsv(OVe[2],24)   # output subspace of L0H2
U3 = lsv(OVe[3],6)
Pm = np.hstack([P,T]); 
print("dims: P 24, T 30, Ub 12, U2 24, U3 6 ; random baselines = dim/384:", np.array([24,30,12,24,6])/384)
print("overlaps: Ub vs P", np.linalg.svd(Ub.T@P,compute_uv=False)[:5], " Ub vs T", np.linalg.svd(Ub.T@T,compute_uv=False)[:5])
print("U2 vs P", np.linalg.svd(U2.T@P,compute_uv=False)[:5], " U2 vs T", np.linalg.svd(U2.T@T,compute_uv=False)[:5], "U2 vs Ub", np.linalg.svd(U2.T@Ub,compute_uv=False)[:5])
F=W["W_fc.0"]*W["ln2_w.0"]
print("W_fc.0: energy in P %.3f T %.3f Ub %.3f U2 %.3f U3 %.3f" % tuple(energy(F,S) for S in [P,T,Ub,U2,U3]))
Pj=W["W_proj.0"]
print("W_proj.0 (rows=out dims): out-energy in P %.3f T %.3f Ub %.3f U2 %.3f" % tuple(energy(Pj.T,S) for S in [P,T,Ub,U2]))
for l in [1,2,3]:
    for j in range(6):
        q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l); v=eff(f"W_v.{l}",j,l)
        print(f"L{l}H{j}  Q: P {energy(q,P):.2f} T {energy(q,T):.2f} Ub {energy(q,Ub):.2f} U2 {energy(q,U2):.2f} |  K: P {energy(k,P):.2f} T {energy(k,T):.2f} Ub {energy(k,Ub):.2f} U2 {energy(k,U2):.2f} |  V: P {energy(v,P):.2f} T {energy(v,T):.2f} Ub {energy(v,Ub):.2f} U2 {energy(v,U2):.2f}")
    F=W[f"W_fc.{l}"]*W[f"ln2_w.{l}"]
    print(f"W_fc.{l}: energy in P %.3f T %.3f Ub %.3f U2 %.3f" % tuple(energy(F,S) for S in [P,T,Ub,U2]))
```