```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g
U,S,Vt=np.linalg.svd(Wq2.T@Wk2)
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]; Pc=Uc[:,60:]   # positional subspace & complement
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1)
print("MLP0 neuron input norms: quantiles", np.quantile(rn,[0,.1,.25,.5,.75,.9,1]), " count>0.8:", (rn>0.8).sum(), " count<0.5:", (rn<0.5).sum())
# output subspaces of each L0 head (W_o columns) orthonormalized
Os=[]
for j in range(H):
    s=slice(j*hd,(j+1)*hd); Q,_=np.linalg.qr(W["W_o.0"][:,s]); Os.append(Q)
fce=fc*g2   # effective read on normalized residual (approx: ignoring that LN2 normalizes a sum)
strong=rn>0.8
def fr(M,Q): return np.linalg.norm(M@Q,axis=1)**2/np.linalg.norm(M,axis=1)**2
print("fraction of MLP0 neuron input energy in each subspace (mean over strong neurons / weak neurons); random expectation 64/384=0.167, P: 0.156")
for j in range(H):
    f=fr(fce,Os[j]); print(f"  L0H{j} out-subspace: strong {f[strong].mean():.3f} weak {f[~strong].mean():.3f}")
f=fr(fce,P); print(f"  P (positional): strong {f[strong].mean():.3f} weak {f[~strong].mean():.3f}")
# joint: neurons reading both token (complement of P and of all head outputs) and L0H2 output
fH2=fr(fce,Os[2]); fP=fr(fce,P)
print("corr between H2-fraction and P-fraction among strong neurons:", np.corrcoef(fH2[strong],fP[strong])[0,1])
print("hist of H2-fraction among strong neurons:", np.histogram(fH2[strong],bins=[0,.05,.1,.2,.3,.4,.6,1])[0])
# Baseline: random rotation of W_o
rng=np.random.default_rng(1)
Q,_=np.linalg.qr(rng.standard_normal((d,d))); 
print("baseline random subspace frac:", fr(fce,Q[:,:64])[strong].mean())
# Also: how much does MLP0 read from the dims with ln1_w.0 high vs low (raw)
# And W_proj.0 output: fraction landing in P, in L0 head output subspaces
for j in range(H):
    print(f"proj0 out frac in L0H{j} out-subspace: {np.linalg.norm(Os[j].T@W['W_proj.0'])**2/np.linalg.norm(W['W_proj.0'])**2:.3f}")
# ln1_b.1 large dims
b1=W["ln1_b.1"]; idx=np.argsort(-np.abs(b1))[:12]
print("ln1_b.1 largest:", idx, b1[idx])
print("ln1_w.1 at those:", W["ln1_w.1"][idx])
# which L0 heads write into those dims
for j in range(H):
    s=slice(j*hd,(j+1)*hd); print(f"  L0H{j} W_o row norms at those dims:", np.linalg.norm(W["W_o.0"][idx][:,s],axis=1))
print("  proj0 row norms at those dims:", np.linalg.norm(W["W_proj.0"][idx],axis=1), " mean all", np.linalg.norm(W["W_proj.0"],axis=1).mean())
```