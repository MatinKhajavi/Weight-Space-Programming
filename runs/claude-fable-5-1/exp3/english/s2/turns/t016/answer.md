```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
Uf,Sf,Vf=np.linalg.svd((fc*g2)[strong],full_matrices=False); T=Vf[:30].T
dirs={}
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wk=W[f"W_k.{l}"][s]*gl; Wq=W[f"W_q.{l}"][s]*gl
        u,sv,vt=np.linalg.svd(Wk@T); dirs[f"L{l}H{j}k"]=vt[0]
        u,sv,vt=np.linalg.svd(Wq@T); dirs[f"L{l}H{j}q"]=vt[0]
keys=[k for k in dirs if k.endswith("k") and k.startswith("L1")]+[k for k in dirs if k.endswith("q") and k.startswith("L1")]
M=np.array([[abs(dirs[a]@dirs[b]) for b in keys] for a in keys])
print("abs cosines among top key/query token-directions (layer 1):")
print("      "+" ".join(f"{k:>6}" for k in keys))
for a,row in zip(keys,M): print(f"{a:>6} "+" ".join(f"{x:6.2f}" for x in row))
# compare with the MLP0 top direction (in T coords it's e_0 since T=Vf[:30]) 
e0=np.zeros(30); e0[0]=1
print("cos with MLP0 top direction e0:", {k: round(float(abs(dirs[k]@e0)),2) for k in keys})
# which token-directions are important: project all heads' k@T, q@T, v@T energy onto e0 vs others
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]
    row=[]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        for nm in ["W_q","W_k","W_v"]:
            M=W[f"{nm}.{l}"][s]*gl
            X=M@T; row.append(np.linalg.norm(X[:,0])**2/np.linalg.norm(X)**2)
    print(f"layer {l} frac of T-energy on e0 (q,k,v per head; uniform=0.033):", np.round(row,2))
# how many MLP0 strong neurons have most of their T-energy on e0?
X=(fc*g2)[strong]@T; f0=X[:,0]**2/np.sum(X**2,axis=1)
print("MLP0 strong neurons: frac of T-energy on e0 quantiles", np.quantile(f0,[.1,.5,.9]), " count>0.5:", (f0>0.5).sum())
print("sign of e0 component among strong neurons: pos", (X[:,0]>0).sum(), "neg", (X[:,0]<0).sum(), "; weighted:", (X[:,0]).sum())
```