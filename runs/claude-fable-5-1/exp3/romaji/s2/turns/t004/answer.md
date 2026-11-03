```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d=384; hd=64
def QK(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_q.{l}'][s].T@W[f'W_k.{l}'][s]
def OV(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,s]@W[f'W_v.{l}'][s]
for l in [1,2,3]:
    Ms=[QK(l,j) for j in range(6)]
    mean=np.mean(Ms,0)
    U,S,Vt=np.linalg.svd(mean)
    print(f"layer {l} mean QK sv", S[:10], "fro", np.linalg.norm(mean))
    print("  sym/asym of mean:", np.linalg.norm(mean+mean.T)/2, np.linalg.norm(mean-mean.T)/2)
    # how much of each head's QK is explained by top-k of mean
    for k in [1,2,4]:
        P=U[:,:k]@U[:,:k].T; Q=Vt[:k].T@Vt[:k]
        print(f"  k={k}: frac of head QK energy in Umean-k rows:", [round(np.linalg.norm(P@M)**2/np.linalg.norm(M)**2,3) for M in Ms],
              " cols:", [round(np.linalg.norm(M@Q)**2/np.linalg.norm(M)**2,3) for M in Ms])
    # bias direction
    b=W[f'ln1_b.{l}']; big=np.argsort(-np.abs(b))[:8]; print("  ln1_b big dims", big, b[big])
    # top left/right singular vectors of mean: their components on big dims
    print("  U0 on big dims", U[big,0], " V0 on big dims", Vt[0,big])
    print("  U0 top dims", np.argsort(-np.abs(U[:,0]))[:8], " V0 top dims", np.argsort(-np.abs(Vt[0]))[:8])
```