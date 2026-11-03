```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
rng=np.random.default_rng(0)
Q1=hm(0,'q',1)*ln0; K1=hm(0,'k',1)*ln0
U,S,Vt=np.linalg.svd(Q1.T@K1)
v0=Vt[0]; v1=Vt[1]; u0=U[:,0]; u1=U[:,1]
readers={'V3':hm(0,'v',3)*ln0,'V0':hm(0,'v',0)*ln0,'V5':hm(0,'v',5)*ln0,'V1':hm(0,'v',1)*ln0,'Fc0':W['W_fc.0']*W['ln2_w.0'],
         'Q3':hm(0,'q',3)*ln0,'K3':hm(0,'k',3)*ln0,'Q0':hm(0,'q',0)*ln0,'K0':hm(0,'k',0)*ln0,'K5':hm(0,'k',5)*ln0,'Q5':hm(0,'q',5)*ln0}
R=rng.standard_normal((384,200)); R/=np.linalg.norm(R,axis=0)
print("reading strength relative to random unit directions:")
for name,A in readers.items():
    base=np.mean(np.linalg.norm(A@R,axis=0))
    print(f" {name}: k-dir0 {np.linalg.norm(A@v0)/base:.2f}  k-dir1 {np.linalg.norm(A@v1)/base:.2f}  q-dir0 {np.linalg.norm(A@u0)/base:.2f}  q-dir1 {np.linalg.norm(A@u1)/base:.2f}")
# Compare k-dir0 of H1 with those of H2,H4,H5 and with ln1_w.0 pattern
print("ln1_w.0 on k-dir0's top dims:", [(int(i), round(float(W['ln1_w.0'][i]),2), round(float(v0[i]),2)) for i in np.argsort(-np.abs(v0))[:12]])
print("ln1_w.0 on k-dir1's top dims:", [(int(i), round(float(W['ln1_w.0'][i]),2), round(float(v1[i]),2)) for i in np.argsort(-np.abs(v1))[:12]])
```