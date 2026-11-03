```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
OV0=W['W_o.0'][:,0:64]@(hm(0,'v',0)*ln0); Uo,So,_=np.linalg.svd(OV0); H0out=Uo[:,:8]
OV3=W['W_o.0'][:,192:256]@(hm(0,'v',3)*ln0); Uo,So,_=np.linalg.svd(OV3); H3out8=Uo[:,:8]
special=[21,39,58,99,184,383]
print("chance 8/384 =",8/384)
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    for name in ['q','k','v']:
        row=[]
        for j in range(6):
            A=(hm(l,name,j)*lnl).copy(); A[:,special]=0
            row.append(f"{np.linalg.norm(A@H0out)**2/np.linalg.norm(A)**2:.3f}/{np.linalg.norm(A@H3out8)**2/np.linalg.norm(A)**2:.3f}")
        print(f"L{l} {name} (H0out8/H3out8):"," ".join(row))
for l in range(4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print(f"MLP{l} in: H0out8 {np.linalg.norm(F@H0out)**2/np.linalg.norm(F)**2:.3f}  H3out8 {np.linalg.norm(F@H3out8)**2/np.linalg.norm(F)**2:.3f}")
# Does anything else write into H0out? e.g., MLP0 output, later heads' outputs
for l in range(4):
    Pj=W[f'W_proj.{l}']
    print(f"W_proj.{l} out frac in H0out8: {np.linalg.norm(H0out.T@Pj)**2/np.linalg.norm(Pj)**2:.3f}, in H3out8: {np.linalg.norm(H3out8.T@Pj)**2/np.linalg.norm(Pj)**2:.3f}")
```