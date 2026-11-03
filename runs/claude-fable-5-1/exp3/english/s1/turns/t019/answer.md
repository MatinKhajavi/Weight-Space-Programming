```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def lsv(A,r):
    u,_,_=np.linalg.svd(A,full_matrices=False); return u[:,:r]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
subs = {}
subs["Mout0"] = lsv(W["W_proj.0"],60)
subs["Mout1"] = lsv(W["W_proj.1"],60)
subs["Mout2"] = lsv(W["W_proj.2"],60)
subs["A1out"] = lsv(np.hstack([OVe[(1,j)] for j in range(6)]),60)
subs["A2out"] = lsv(np.hstack([OVe[(2,j)] for j in range(6)]),60)
def stats(M):
    ev=np.linalg.eigvals(M); r=ev.real.sum()/np.abs(ev).sum()
    sym=np.linalg.norm(M+M.T)/2; asym=np.linalg.norm(M-M.T)/2
    w=np.linalg.eigvalsh((M+M.T)/2)
    return r, sym/asym, w.max(), w.min()
for l in [1,2,3]:
    for j in range(6):
        q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l); B=q.T@k/8
        out=[]
        for nm,S in subs.items():
            if (nm=="Mout1" or nm=="A1out") and l<2: continue
            if (nm=="Mout2" or nm=="A2out") and l<3: continue
            M=S.T@B@S
            en=np.linalg.norm(M)**2/np.linalg.norm(B)**2
            r,sa,wmax,wmin=stats(M)
            out.append(f"{nm}: en {en:.2f} realfrac {r:+.2f} sym/asym {sa:.2f} symeig[{wmax:+.2f},{wmin:+.2f}]")
        print(f"L{l}H{j} | "+" | ".join(out))
# Also: V of layers 2,3: energy in those subspaces (what they copy)
for l in [2,3]:
    for j in range(6):
        v=eff(f"W_v.{l}",j,l)
        print(f"L{l}H{j} V energy:", {nm: round(float(np.linalg.norm(v@S)**2/np.linalg.norm(v)**2),2) for nm,S in subs.items()})
```