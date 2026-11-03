```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
special=[21,39,58,99,184,383]
A=W['W_proj.3'].copy(); A[special,:]=0
Up,Sp,_=np.linalg.svd(A); L=Up[:,:24]
V3=hm(0,'v',3)*ln0; _,_,VtV=np.linalg.svd(V3)
Fc=W['W_fc.0']*W['ln2_w.0']; _,_,VtF=np.linalg.svd(Fc)
OV3=W['W_o.0'][:,192:256]@V3
OVc=W['W_o.3'][:,192:256]@(hm(3,'v',3)*W['ln1_w.3'])   # copy head
Mlin0=0.5*W['W_proj.0']@Fc
rng=np.random.default_rng(0)
def align(C,M): return float(np.sum(C*M)/np.linalg.norm(C)/np.linalg.norm(M))
for name,Vt in [('V3',VtV),('Fc0',VtF)]:
    for k in [16,24,32]:
        T=Vt[:k].T
        C=L.T@OVc@T                     # token emb -> same-token logit (copy head)
        Mprev_direct=L.T@OV3@T          # prev token -> logit (direct path via H3)
        Mprev_mlp=L.T@Mlin0@OV3@T       # prev token -> logit via MLP0 (linearized)
        Mcur_mlp=L.T@Mlin0@T            # current token -> logit via MLP0
        Mcur_direct=L.T@T               # current token embedding directly in logit subspace
        # baselines: random rotations of C
        bl=[align(L.T@(np.linalg.qr(rng.standard_normal((384,384)))[0]@OVc)@T, Mprev_mlp) for _ in range(10)]
        print(f"T={name} k={k}: align(copy, prev->logit direct)={align(C,Mprev_direct):+.3f}  align(copy, prev->logit MLP0)={align(C,Mprev_mlp):+.3f}  align(copy, cur->logit MLP0)={align(C,Mcur_mlp):+.3f}  align(copy, cur direct)={align(C,Mcur_direct):+.3f}   baseline {np.mean(bl):+.3f}±{np.std(bl):.3f}")
```