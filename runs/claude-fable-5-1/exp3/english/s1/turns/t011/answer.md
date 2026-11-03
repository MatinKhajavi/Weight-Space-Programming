```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
def lsv(A,r):
    u,_,_=np.linalg.svd(A,full_matrices=False); return u[:,:r]
Q2=eff("W_q.0",2); K2=eff("W_k.0",2)
for r in [16,24,40]:
    P = rsv(np.vstack([Q2,K2]),r)
    M = P.T@Q2.T@K2@P/8
    ev = np.linalg.eigvals(M)
    idx=np.argsort(-np.abs(ev))
    print(f"L0H2 QK form on P({r}): eigenvalues (sorted by |.|):")
    print("  re:", ev[idx].real[:20]); print("  im:", ev[idx].imag[:20])
    print("  real-frac %.2f ; sym part norm %.2f asym part norm %.2f" % (ev.real.sum()/np.abs(ev).sum(), np.linalg.norm(M+M.T)/2, np.linalg.norm(M-M.T)/2))
# same for L0H3 and the broad heads (their own top subspaces)
for j in [0,1,3,4,5]:
    q=eff("W_q.0",j); k=eff("W_k.0",j)
    Pj = rsv(np.vstack([q,k]),16)
    M = Pj.T@q.T@k@Pj/8
    ev=np.linalg.eigvals(M); idx=np.argsort(-np.abs(ev))
    print(f"L0H{j}: top eig re {ev[idx].real[:6]} im {ev[idx].imag[:6]}  real-frac {ev.real.sum()/np.abs(ev).sum():+.2f}")
# L3H1 induction-like form
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
E = rsv(Fcomp,30)
for (l,j) in [(3,1),(3,5),(1,5),(2,2)]:
    q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l)
    M = E.T@q.T@k@OVe[(0,2)]@E/8
    ev=np.linalg.eigvals(M); idx=np.argsort(-np.abs(ev))
    print(f"L{l}H{j} via L0H2 on E: eig re {ev[idx].real[:10]} im {ev[idx].imag[:10]}")
    Ms=(M+M.T)/2; w=np.linalg.eigvalsh(Ms); print("   sym-part eigs:", np.sort(w)[::-1][:8], "...", np.sort(w)[:4])
```