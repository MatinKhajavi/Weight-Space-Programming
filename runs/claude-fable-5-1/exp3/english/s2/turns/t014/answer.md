```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g; Wv2=W["W_v.0"][s2]*g
U,S,Vt=np.linalg.svd(Wq2.T@Wk2)
for r in [40,60]:
    Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:r],Vt[:r].T],axis=1)); P=Uc[:,:r]
    Pc=np.linalg.svd(P.T,full_matrices=True)[2][r:].T  # complement basis
    sv=np.linalg.svd(Wv2@Pc,compute_uv=False)
    print(f"P rank {r}: L0H2 V in P-complement sv:", sv[:40])
    sv3=np.linalg.svd((W['W_v.0'][192:256]*g)@Pc,compute_uv=False); print("   L0H3 V in P-complement sv:", sv3[:36])
    sv0=np.linalg.svd((W['W_v.0'][0:64]*g)@Pc,compute_uv=False); print("   L0H0 V in P-complement sv:", sv0[:36])
    fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
    svf=np.linalg.svd((fc*g2)[strong]@Pc,compute_uv=False); print("   MLP0 strong in P-complement sv:", svf[:45])
    svfw=np.linalg.svd((fc*g2)[~strong]@Pc,compute_uv=False); print("   MLP0 weak in P-complement sv:", svfw[:10])
```