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
Ub,_ = np.linalg.qr(np.hstack([lsv(OVe[j],3) for j in [0,1,4,5]]))
U2 = lsv(OVe[2],24)
F = W["W_fc.0"]*W["ln2_w.0"]
Fs = np.linalg.svd(F,compute_uv=False)
F0 = rsv(F,30)
print("F0 vs P", np.linalg.svd(F0.T@P,compute_uv=False)[:6])
print("F0 vs T", np.linalg.svd(F0.T@T,compute_uv=False)[:6])
print("F0 vs Ub", np.linalg.svd(F0.T@Ub,compute_uv=False)[:6])
print("F0 vs U2", np.linalg.svd(F0.T@U2,compute_uv=False)[:6])
# complement of P+T+Ub+U2
Sall,_ = np.linalg.qr(np.hstack([P,T,Ub,U2]))
print("dim of union subspace", Sall.shape[1], " W_fc.0 energy in union %.2f" % energy(F,Sall))
# residual reading subspace of MLP0: project out union, see spectrum
Fres = F - (F@Sall)@Sall.T
print("W_fc.0 residual sv:", np.linalg.svd(Fres,compute_uv=False)[:30])
R = rsv(Fres,30)  # 'other' subspace read by MLP0
print("R energy on ln1_w.0 small dims (<0.15): %.2f  (fraction of dims %.2f)" % ((R[w0<0.15]**2).sum()/(R**2).sum(), (w0<0.15).mean()))
print("P on small dims %.2f, T on small dims %.2f, Ub %.2f, U2 %.2f" % tuple((S[w0<0.15]**2).sum()/(S**2).sum() for S in [P,T,Ub,U2]))
# who reads R?
for l in range(4):
    for j in range(6):
        q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l); v=eff(f"W_v.{l}",j,l)
        print(f"L{l}H{j} energy in R: Q {energy(q,R):.2f} K {energy(k,R):.2f} V {energy(v,R):.2f}", end=" | ")
    print()
    Fl = W[f"W_fc.{l}"]*W[f"ln2_w.{l}"]
    print(f"W_fc.{l} in R: {energy(Fl,R):.2f}")
# raw W_fc.0 (no ln2 scaling) energies
Fr = W["W_fc.0"]
print("raw W_fc.0 energy: P %.3f T %.3f Ub %.3f U2 %.3f R %.3f" % tuple(energy(Fr,S) for S in [P,T,Ub,U2,R]))
print("ln2_w.0 mean on dims where R is big:", W["ln2_w.0"][np.argsort(-(R**2).sum(1))[:40]].mean(), " overall", W["ln2_w.0"].mean())
```