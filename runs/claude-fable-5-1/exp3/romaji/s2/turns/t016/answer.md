```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
Q3=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),40)
QK3=blk('W_q',0,3).T@blk('W_k',0,3)
M=Q3.T@QK3@Q3
ev=np.linalg.eigvals(M); idx=np.argsort(-np.abs(ev))
print("L0H3 QK restricted to Q3(40): eigenvalues |λ| and angle(deg):")
for e in ev[idx][:40]: print(f"  |λ|={abs(e):.2f} angle={np.degrees(np.angle(e)):.0f}")
S=np.linalg.svd(M,compute_uv=False); print("sv of restricted:", S)
# symmetric part eigenvalues
Ms=(M+M.T)/2; print("sym part eigs:", np.sort(np.linalg.eigvalsh(Ms))[::-1])
Ma=(M-M.T)/2; print("antisym part sv:", np.linalg.svd(Ma,compute_uv=False)[:20])
# Same for L0H0 in its own q/k subspace
Q0=rowspace(np.vstack([blk('W_q',0,0),blk('W_k',0,0)]),30)
M0=Q0.T@(blk('W_q',0,0).T@blk('W_k',0,0))@Q0
ev=np.linalg.eigvals(M0); idx=np.argsort(-np.abs(ev))
print("L0H0 restricted eigs:", [(round(abs(e),2), int(np.degrees(np.angle(e)))) for e in ev[idx][:20]])
print("L0H0 sym eigs:", np.sort(np.linalg.eigvalsh((M0+M0.T)/2))[::-1][:10], np.sort(np.linalg.eigvalsh((M0+M0.T)/2))[:5])
```