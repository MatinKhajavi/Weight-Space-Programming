```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
ln=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_{name}.{l}'][s]
Q3=hm(0,'q',3)*ln; K3=hm(0,'k',3)*ln
U,S,Vt=np.linalg.svd(np.vstack([Q3,K3]))
print("sv of [Q3;K3]:",S[:40])
for k in [8,12,16,20,24,32,48]:
    P=Vt[:k]  # k x 384
    print(f"--- k={k}: expected frac {k/384:.3f}")
    res=[]
    for l in [0]:
        for name in ['q','k','v']:
            for j in range(6):
                A=hm(0,name,j)*ln
                fr=np.linalg.norm(A@P.T)**2/np.linalg.norm(A)**2
                res.append(f"{name}{j}:{fr:.2f}")
    print("  L0 (ln folded):"," ".join(res))
    A=W['W_fc.0']*W['ln2_w.0']; print(f"  W_fc.0 (ln2 folded): {np.linalg.norm(A@P.T)**2/np.linalg.norm(A)**2:.3f}")
    for l in [1,2,3]:
        lnl=W[f'ln1_w.{l}']
        res=[]
        for name in ['q','k','v']:
            for j in range(6):
                A=hm(l,name,j)*lnl
                A=A.copy(); A[:,[21,39,58,99,184,383]]=0
                fr=np.linalg.norm(A@P.T)**2/np.linalg.norm(A)**2
                res.append(f"{name}{j}:{fr:.2f}")
        print(f"  L{l}:"," ".join(res))
# which dims dominate P (k=20)
P=Vt[:20]
w=np.linalg.norm(P,axis=0)**2
print("dims most in positional subspace:", np.argsort(-w)[:40], np.sort(w)[::-1][:40])
```