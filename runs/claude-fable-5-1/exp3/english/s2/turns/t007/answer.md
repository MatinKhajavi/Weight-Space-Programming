```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g
U,S,Vt=np.linalg.svd(Wq2.T@Wk2)
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
fce=(fc*g2)[strong]
Uf,Sf,Vf=np.linalg.svd(fce,full_matrices=False)
T=Vf[:30].T
def heads(l):
    out=[]; gl=W[f"ln1_w.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv,Wo=W[f"W_q.{l}"][s]*gl,W[f"W_k.{l}"][s]*gl,W[f"W_v.{l}"][s]*gl,W[f"W_o.{l}"][:,s]
        out.append(dict(QK=Wq.T@Wk,OV=Wo@Wv))
    return out
HS=[heads(l) for l in range(4)]
M=T.T@HS[3][1]["QK"]@HS[0][2]["OV"]@T
print("L3H1 induction matrix diag:", np.diag(M))
print("  offdiag rms %.3f, diag mean %.3f"%(np.sqrt((M**2).sum()-(np.diag(M)**2).sum())/np.sqrt(M.size-30), np.diag(M).mean()))
ev=np.linalg.eigvals((M+M.T)/2); print("  sym eigs:", np.sort(ev)[::-1][:8], np.sort(ev)[:4])
for j in range(H):
    M=T.T@HS[1][j]["QK"]@T
    print(f"L1H{j} same-token matrix diag:", np.diag(M)[:12], " offdiag rms %.3f"%(np.sqrt(((M-np.diag(np.diag(M)))**2).sum()/(M.size-30))))
M=T.T@HS[3][1]["QK"]@T; print("L3H1 same-token diag:", np.diag(M)[:12])
M=T.T@HS[3][0]["QK"]@T; print("L3H0 same-token diag:", np.diag(M)[:12])
# Which directions are these? direction 0 of T: is it in P? is it a 'constant' direction (large ln biases)? 
t0=T[:,0]
print("t0 in P frac:", np.linalg.norm(P.T@t0)**2)
# check whether t0 aligns with output directions of broad heads (their mean output) - compute W_o columns of L0H0 projected
for j in range(H):
    s=slice(j*hd,(j+1)*hd); Wo=W["W_o.0"][:,s]; print(f"  |t0^T Wo L0H{j}| / |Wo| = {np.linalg.norm(t0@Wo)/np.linalg.norm(Wo):.3f} (random {np.sqrt(1/384):.3f})")
print("  proj0:", np.linalg.norm(t0@W["W_proj.0"])/np.linalg.norm(W["W_proj.0"]))
# repeat tests excluding top-1 and top-2 directions of T
for drop in [1,2,3]:
    T2=Vf[drop:30].T
    def idscore(M): n=M.shape[0]; return np.trace(M)/np.sqrt(n)/np.linalg.norm(M)
    print(f"drop {drop}: same-token L1:", [round(idscore(T2.T@HS[1][j]["QK"]@T2),2) for j in range(H)], " L3:", [round(idscore(T2.T@HS[3][j]["QK"]@T2),2) for j in range(H)])
    print(f"   induction via L0H2 L3:", [round(idscore(T2.T@HS[3][j]["QK"]@HS[0][2]["OV"]@T2),2) for j in range(H)], " L2:", [round(idscore(T2.T@HS[2][j]["QK"]@HS[0][2]["OV"]@T2),2) for j in range(H)], " L1:", [round(idscore(T2.T@HS[1][j]["QK"]@HS[0][2]["OV"]@T2),2) for j in range(H)])
    print(f"   induction via L0H3 L3:", [round(idscore(T2.T@HS[3][j]["QK"]@HS[0][3]["OV"]@T2),2) for j in range(H)])
    print(f"   OV self L2:", [round(idscore(T2.T@HS[2][j]["OV"]@T2),2) for j in range(H)], " L3:", [round(idscore(T2.T@HS[3][j]["OV"]@T2),2) for j in range(H)], " L0:", [round(idscore(T2.T@HS[0][j]["OV"]@T2),2) for j in range(H)])
```