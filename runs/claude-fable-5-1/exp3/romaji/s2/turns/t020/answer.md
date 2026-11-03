```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
T3=rowspace(blk('W_v',0,3),24); Q3=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),30)
fc=W['W_fc.0']; A=fc[np.linalg.norm(fc,axis=1)>1]
U,S,Vt=np.linalg.svd(A,full_matrices=False)
keys={}
for j in [1,2,4,5]:
    QK=blk('W_q',0,j).T@blk('W_k',0,j); u,s,vt=np.linalg.svd(QK); keys[j]=vt[0]
for r in range(6):
    v=Vt[r]
    print(f"fc.0 dir{r} sv={S[r]:.1f}: in T3 {np.linalg.norm(v@T3)**2:.2f}, in Q3 {np.linalg.norm(v@Q3)**2:.2f}; cos with L0 key dirs H1 {v@keys[1]:+.2f} H2 {v@keys[2]:+.2f} H4 {v@keys[4]:+.2f} H5 {v@keys[5]:+.2f}; cos with b0 {v@W['ln1_b.0']/np.linalg.norm(W['ln1_b.0']):+.2f}")
# how many neurons load strongly on dir0 and what is the sign distribution
proj=A@Vt[0]; print("neuron loadings on dir0: pct", np.percentile(proj,[1,10,50,90,99]))
proj=A@Vt[1]; print("neuron loadings on dir1: pct", np.percentile(proj,[1,10,50,90,99]))
# Also: W_v.0 of H5 read directions vs fc dirs
u,s,vt=np.linalg.svd(blk('W_v',0,5))
for r in range(4): print(f"H5 v dir{r}: cos with fc dirs 0-5:", np.round(vt[r]@Vt[:6].T,2))
```