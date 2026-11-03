```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
M={}
for j in range(6):
    Q=hm(0,'q',j)*ln0; K=hm(0,'k',j)*ln0; M[j]=Q.T@K/8
M3=M[3]
# normalize M3 so that M3^T M3 ~ projector: scale by its typical singular value
s=np.linalg.svd(M3,compute_uv=False); sc=s[:16].mean(); S=M3/sc
basis={'S':S,'S^T':S.T,'S^2':S@S,'(S^T)^2':S.T@S.T,'S^T S':S.T@S,'S S^T':S@S.T,'S^3':S@S@S,'(S^T)^3':S.T@S.T@S.T}
names=list(basis.keys())
B=np.stack([basis[n].ravel() for n in names],1)
for j in [0,1,2,4,5]:
    y=M[j].ravel()
    coef,res,rk,sv=np.linalg.lstsq(B,y,rcond=None)
    pred=B@coef
    r2=1-np.sum((y-pred)**2)/np.sum(y**2)
    print(f"L0H{j}: R^2={r2:.3f}; coefs:", {n: round(float(c),3) for n,c in zip(names,coef)})
# sanity: M3 itself
y=M3.ravel(); coef,_,_,_=np.linalg.lstsq(B,y,rcond=None); print("M3 self-fit coefs:", {n: round(float(c),3) for n,c in zip(names,coef)})
# How 'shift-like' is S: compare S^T S with S S^T (should both be ~projector P) and check S^2 norm
print("|S|=%.2f |S^2|=%.2f |S^3|=%.2f |S^T S|=%.2f"%(np.linalg.norm(S),np.linalg.norm(S@S),np.linalg.norm(S@S@S),np.linalg.norm(S.T@S)))
```