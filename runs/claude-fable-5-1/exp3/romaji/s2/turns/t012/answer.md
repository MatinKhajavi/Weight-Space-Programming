```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
T3=rowspace(blk('W_v',0,3),24)
keys={}; qs={}
for j in [0,1,2,4,5]:
    QK=blk('W_q',0,j).T@blk('W_k',0,j); U,S,Vt=np.linalg.svd(QK)
    for r in range(3):
        keys[(j,r)]=Vt[r]*S[r]; qs[(j,r)]=U[:,r]*S[r]
names=list(keys.keys())
print("cos between key directions (weighted by sv) of L0 heads comps:")
M=np.array([[ (keys[a]@keys[b])/np.linalg.norm(keys[a])/np.linalg.norm(keys[b]) for b in names] for a in names])
print(names); print(M)
print("cos between query directions:")
M=np.array([[ (qs[a]@qs[b])/np.linalg.norm(qs[a])/np.linalg.norm(qs[b]) for b in names] for a in names]); print(M)
# key dirs projected on T3 then compare
print("key dirs projected on T3(24): cos")
kp={a:T3.T@keys[a] for a in names}
M=np.array([[ (kp[a]@kp[b])/np.linalg.norm(kp[a])/np.linalg.norm(kp[b]) for b in names] for a in names]); print(M)
print("norm fraction in T3:", [round(np.linalg.norm(kp[a])**2/np.linalg.norm(keys[a])**2,2) for a in names])
# Which L0 heads' key directions are also strongly read by fc.0?  (cosine of key dir with fc.0 rows: max |cos|)
fc=W['W_fc.0']; fcn=fc/np.linalg.norm(fc,axis=1,keepdims=True)
for a in names:
    c=fcn@(keys[a]/np.linalg.norm(keys[a])); print(a, "fc.0 rows with |cos|>0.5:", (np.abs(c)>0.5).sum(), " max", np.abs(c).max().round(2))
```