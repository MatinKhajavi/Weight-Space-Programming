```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=W['ln1_w.0']
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def colspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return U[:,:k]
OV3=P@W['W_o.0'][:,192:256]@blk('W_v',0,3)@np.diag(w0)
C3=colspace(OV3,20)
OV0=P@W['W_o.0'][:,0:64]@blk('W_v',0,0)@np.diag(w0); C0=colspace(OV0,20)
Cp=colspace(np.hstack([P@W['W_o.0'][:,a*hd:(a+1)*hd]@blk('W_v',0,a)@np.diag(w0) for a in [1,2,4,5]]),20)
Cm=colspace(P@W['W_proj.0'],40)
def frac(A,B): return np.linalg.norm(A@B)**2/np.linalg.norm(A)**2
print("fraction of read energy in: L0H3-out(20) [rand .052] | L0H0-out(20) | L0H1245-out(20) | MLP0-out(40) [rand .104]")
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        for nm in ['W_q','W_k','W_v']:
            R=blk(nm,l,j)@wl
            print(f"L{l}H{j} {nm[2]}: {frac(R,C3):.3f} {frac(R,C0):.3f} {frac(R,Cp):.3f} {frac(R,Cm):.3f}", end=" | ")
        print()
    R=W[f'W_fc.{l}']@np.diag(W[f'ln2_w.{l}'])
    print(f"MLP{l}: {frac(R,C3):.3f} {frac(R,C0):.3f} {frac(R,Cp):.3f} {frac(R,Cm):.3f}")
```