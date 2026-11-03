```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T, S
# token subspace estimate from L0H3's V reader
T3,S=rowspace(blk('W_v',0,3),20); print("L0H3 v sv", S[:30])
def frac(A,B):  # fraction of row-space energy of A (rows) inside subspace B (columns orthonormal)
    return np.linalg.norm(A@B)**2/np.linalg.norm(A)**2
for j in range(6):
    print(f"L0H{j}: frac in T3(20): q={frac(blk('W_q',0,j),T3):.2f} k={frac(blk('W_k',0,j),T3):.2f} v={frac(blk('W_v',0,j),T3):.2f}")
print("fc.0 frac in T3:", frac(W['W_fc.0'],T3), " (20/384 random expectation =", 20/384,")")
# Now subspace read by L0H3 q and k
Q3,S=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),30); print("L0H3 qk stacked sv", S[:40])
for j in range(6):
    print(f"L0H{j}: frac in Q3(30): q={frac(blk('W_q',0,j),Q3):.2f} k={frac(blk('W_k',0,j),Q3):.2f} v={frac(blk('W_v',0,j),Q3):.2f}")
print("fc.0 frac in Q3:", frac(W['W_fc.0'],Q3), " random exp", 30/384)
# overlap between T3 and Q3
print("principal cosines T3 vs Q3:", np.linalg.svd(T3.T@Q3,compute_uv=False)[:20])
# key directions of H1,H2,H4 and H5 and their projection
for j in [1,2,4,5,0]:
    QK=blk('W_q',0,j).T@blk('W_k',0,j); U,S,Vt=np.linalg.svd(QK)
    for r in range(3):
        u=U[:,r]; v=Vt[r]
        print(f"L0H{j} comp{r} sv={S[r]:.2f}: query dir in T3 {np.linalg.norm(u@T3)**2:.2f} in Q3 {np.linalg.norm(u@Q3)**2:.2f}; key dir in T3 {np.linalg.norm(v@T3)**2:.2f} in Q3 {np.linalg.norm(v@Q3)**2:.2f}; u.v={u@v:.2f}")
# the ov read directions of H5
U,S,Vt=np.linalg.svd(blk('W_v',0,5))
for r in range(4): print(f"L0H5 v dir{r} sv={S[r]:.2f}: in T3 {np.linalg.norm(Vt[r]@T3)**2:.2f} in Q3 {np.linalg.norm(Vt[r]@Q3)**2:.2f}")
for j in [1,2,4]:
    U,S,Vt=np.linalg.svd(blk('W_v',0,j))
    for r in range(3): print(f"L0H{j} v dir{r} sv={S[r]:.2f}: in T3 {np.linalg.norm(Vt[r]@T3)**2:.2f} in Q3 {np.linalg.norm(Vt[r]@Q3)**2:.2f}")
```