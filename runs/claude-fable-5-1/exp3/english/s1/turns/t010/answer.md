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
P = rsv(np.vstack([Q2,K2]),24)
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
E = rsv(Fcomp,30)   # token subspace estimate
def eigstats(M):
    ev = np.linalg.eigvals(M)
    return ev.real.sum()/np.abs(ev).sum(), np.linalg.norm(M+M.T)/np.linalg.norm(M-M.T)
print("== OV restricted to token subspace E: E^T OV E : real-sum/abs-sum, sym/asym norm ratio, |E^T OV E|/|OV| energy")
for l in range(4):
    for j in range(6):
        M = E.T@OVe[(l,j)]@E
        r,sa = eigstats(M)
        print(f"L{l}H{j}: eig-real-frac {r:+.2f}  sym/asym {sa:.2f}  energy E->E {np.linalg.norm(M)**2/np.linalg.norm(OVe[(l,j)])**2:.3f} (baseline {30*30/384/384:.3f})", end="   ")
    print()
print("== induction test: E^T Q_l^T K_l OV_0h E for later heads, with each layer-0 head's OV (ratio of Frobenius norm to random-rotation baseline, and eig-real-frac)")
rng=np.random.default_rng(0)
for l in [1,2,3]:
    for j in range(6):
        q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l)
        B = q.T@k
        out=[]
        for h in range(6):
            M = E.T@B@OVe[(0,h)]@E
            base=[]
            for _ in range(3):
                Qr,_=np.linalg.qr(rng.normal(size=(384,384)))
                base.append(np.linalg.norm(E.T@B@Qr@OVe[(0,h)]@E))
            r,_=eigstats(M)
            out.append((np.linalg.norm(M)/np.mean(base), r))
        print(f"L{l}H{j}:", " ".join(f"h{h}:{a:.1f}x/{b:+.2f}" for h,(a,b) in enumerate(out)))
# also direct token-token form of each head (Q,K both reading E): E^T Q^T K E
print("== direct token-token QK form E^T Q^T K E: energy fraction and eig-real-frac")
for l in range(4):
    for j in range(6):
        q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l); B=q.T@k
        M=E.T@B@E; r,sa=eigstats(M)
        print(f"L{l}H{j}: {np.linalg.norm(M)**2/np.linalg.norm(B)**2:.3f} eig {r:+.2f}", end="  ")
    print()
```