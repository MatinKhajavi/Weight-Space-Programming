```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
U,S,Vt=np.linalg.svd((W["W_q.0"][s2]*g).T@(W["W_k.0"][s2]*g))
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
Pc=np.linalg.svd(P.T,full_matrices=True)[2][60:].T  # 324-dim complement
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
def rowspace(M,k):
    u,s,vt=np.linalg.svd(M@Pc,full_matrices=False); return vt[:k].T, s
A,sa=rowspace((fc*g2)[strong],60)
B,sb=rowspace(W["W_v.0"][s2]*g,60)
C,sc=rowspace(np.concatenate([W[f"W_v.1"][j*hd:(j+1)*hd]*W["ln1_w.1"] for j in range(H)]),60)   # layer1 V (all heads) - reads residual incl. attn outputs
D,sd=rowspace(np.concatenate([W[f"W_q.1"][j*hd:(j+1)*hd]*W["ln1_w.1"] for j in range(H)]),60)
for k in [30,40,50]:
    cosAB=np.linalg.svd(A[:,:k].T@B[:,:k],compute_uv=False)
    cosAC=np.linalg.svd(A[:,:k].T@C[:,:k],compute_uv=False)
    cosAD=np.linalg.svd(A[:,:k].T@D[:,:k],compute_uv=False)
    print(f"k={k}: MLP0-in vs L0H2-V principal cosines:", np.round(cosAB,2))
    print(f"       MLP0-in vs L1-V:", np.round(cosAC,2))
    print(f"       MLP0-in vs L1-Q:", np.round(cosAD,2))
    print("   counts >0.8:", (cosAB>0.8).sum(), (cosAC>0.8).sum(), (cosAD>0.8).sum(), " >0.6:", (cosAB>0.6).sum(), (cosAC>0.6).sum(), (cosAD>0.6).sum())
```