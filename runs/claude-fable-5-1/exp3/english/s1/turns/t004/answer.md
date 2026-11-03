```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rng = np.random.default_rng(1)
def OV(l,j):
    s=slice(64*j,64*(j+1)); return W[f"W_o.{l}"][:,s] @ (W[f"W_v.{l}"][s]*W[f"ln1_w.{l}"])
def reader(l,j,which):
    s=slice(64*j,64*(j+1)); return W[f"W_{which}.{l}"][s]*W[f"ln1_w.{l}"]
def comp(B,A):
    return np.linalg.norm(B@A)/(np.linalg.norm(B)*np.linalg.norm(A))
# random baseline: random orthogonal rotation of A's output space
def baseline(B,A,n=3):
    out=[]
    for _ in range(n):
        Q,_=np.linalg.qr(rng.normal(size=(384,384)))
        out.append(comp(B,Q@A))
    return np.mean(out)
for lB in [1,2,3]:
    for lA in range(lB):
        for which in ["q","k","v"]:
            print(f"--- {which.upper()}-composition: layer {lA} heads (cols) -> layer {lB} heads (rows); ratio to random-rotation baseline")
            for jB in range(6):
                B = reader(lB,jB,which)
                row=[]
                for jA in range(6):
                    A = OV(lA,jA)
                    row.append(comp(B,A)/baseline(B,A))
                print(f"L{lB}H{jB}", np.array(row))
```