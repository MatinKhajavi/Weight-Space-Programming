```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
w=W["ln1_w.0"]; b=W["ln1_b.0"]
idx=np.argsort(-w)
print("ln1_w.0 sorted:", w[idx][:120])
print("count >0.5:", (w>0.5).sum(), " >0.2:", (w>0.2).sum(), " <0.05:", (w<0.05).sum())
hi=np.where(w>0.5)[0]; lo=np.where(w<0.05)[0]
print("hi dims:", hi)
for name in ["W_q.0","W_k.0","W_v.0"]:
    M=W[name]
    cn=np.linalg.norm(M,axis=0)
    print(name, "col norm hi dims mean %.3f, lo dims mean %.3f (init expect %.3f)"%(cn[hi].mean(),cn[lo].mean(),0.02*np.sqrt(384)))
    for j in range(H):
        s=slice(j*hd,(j+1)*hd); cn=np.linalg.norm(M[s],axis=0)
        print("   head",j,"hi %.3f lo %.3f"%(cn[hi].mean(),cn[lo].mean()))
w2=W["ln2_w.0"]; idx2=np.argsort(-w2)
print("ln2_w.0 sorted:", w2[idx2][:40], "...", w2[idx2][-60:])
print("ln2_w.0 on hi dims:", w2[hi])
print("ln2_w.0 on lo dims mean %.3f"%w2[lo].mean())
fc=W["W_fc.0"]; cn=np.linalg.norm(fc,axis=0)
print("W_fc.0 col norm on hi dims %.3f lo dims %.3f ; init expect %.3f"%(cn[hi].mean(),cn[lo].mean(),0.02*np.sqrt(1536)))
# are ln1_w hi dims also special in later layers?
for l in range(1,4):
    for nm in ["ln1_w","ln2_w"]:
        v=W[f"{nm}.{l}"]; print(l,nm,"hi %.3f lo %.3f"%(v[hi].mean(),v[lo].mean()))
# W_o.0 / W_proj.0 output rows: which dims do layer0 write to
for nm in ["W_o.0","W_proj.0","W_o.1","W_proj.1","W_o.3","W_proj.3"]:
    rn=np.linalg.norm(W[nm],axis=1); print(nm,"row norm hi %.3f lo %.3f"%(rn[hi].mean(),rn[lo].mean()))
```