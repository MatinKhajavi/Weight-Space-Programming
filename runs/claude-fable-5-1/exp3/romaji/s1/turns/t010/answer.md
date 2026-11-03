```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
ln2=W['ln2_w.0']
Fc=W['W_fc.0']*ln2
tot=np.linalg.norm(Fc)**2
print("W_fc.0 total sq norm", tot)
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*W['ln1_w.0'])
    U,S,Vt=np.linalg.svd(OV)
    for k in [2,4,8,16,32]:
        Uk=U[:,:k]
        fr=np.linalg.norm(Fc@Uk)**2/tot
        print(f"L0H{j} out top-{k} dirs: frac of W_fc.0 = {fr:.3f} (chance {k/384:.3f}, ratio {fr/(k/384):.1f}x)", end=" | ")
    print()
# Also the positional subspace P
Q3=W['W_q.0'][192:256]*W['ln1_w.0']; K3=W['W_k.0'][192:256]*W['ln1_w.0']
U,S,Vt=np.linalg.svd(np.vstack([Q3,K3]))
for k in [8,16,24,32]:
    P=Vt[:k]; print(f"pos subspace k={k}: frac {np.linalg.norm(Fc@P.T)**2/tot:.3f} chance {k/384:.3f}")
# The special dims
for d in [21,39,58,99,184,383,163,153]:
    print(f"dim {d}: W_fc.0 column sq norm frac {np.linalg.norm(Fc[:,d])**2/tot:.4f} (chance {1/384:.4f}), ln2_w={ln2[d]:.2f}, ln2_b={W['ln2_b.0'][d]:.2f}")
# Where does W_fc.0 read from most? top dims
cn=np.linalg.norm(Fc,axis=0)
print("W_fc.0 top read dims:", np.argsort(-cn)[:30])
print("values:", np.sort(cn)[::-1][:30])
print("ln2_w.0 top dims:", np.argsort(-ln2)[:30])
```