```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g
U,S,Vt=np.linalg.svd(Wq2.T@Wk2)
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
fce=(fc*g2)[strong]
Uf,Sf,Vf=np.linalg.svd(fce,full_matrices=False)
print("MLP0 strong-neuron input SVD:", Sf[:50])
print("ratio to total energy cum:", (np.cumsum(Sf**2)/np.sum(Sf**2))[[9,19,27,29,39,49,79]])
T=Vf[:30].T
print("overlap of T(30) with P(60): principal cosines", np.linalg.svd(T.T@P,compute_uv=False)[:10])
# L0 heads v reading T vs P
def frac(M,Q): return np.linalg.norm(M@Q)**2/np.linalg.norm(M)**2
for j in range(H):
    s=slice(j*hd,(j+1)*hd)
    print(f"L0H{j}: v in T {frac(W['W_v.0'][s]*g,T):.2f}  v in P {frac(W['W_v.0'][s]*g,P):.2f}; q in T {frac(W['W_q.0'][s]*g,T):.2f} k in T {frac(W['W_k.0'][s]*g,T):.2f}")
# induction / same-token tests
def heads(l):
    out=[]
    gl=W[f"ln1_w.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv,Wo=W[f"W_q.{l}"][s]*gl,W[f"W_k.{l}"][s]*gl,W[f"W_v.{l}"][s]*gl,W[f"W_o.{l}"][:,s]
        out.append(dict(QK=Wq.T@Wk,OV=Wo@Wv))
    return out
HS=[heads(l) for l in range(4)]
def idscore(M):
    n=M.shape[0]; return np.trace(M)/np.sqrt(n)/np.linalg.norm(M)   # 1 if M = cI (c>0), -1 if c<0, ~0 random
print("same-token test T^T QK T (id-score) per head:")
for l in range(4):
    print(" layer",l, [round(idscore(T.T@HS[l][j]["QK"]@T),3) for j in range(H)])
print("induction test T^T QK_l OV_L0Hs T (id-score), rows: L0 source head")
for l in range(1,4):
    for src in range(H):
        print(f" layer {l} via L0H{src}:", [round(idscore(T.T@HS[l][j]["QK"]@HS[0][src]["OV"]@T),3) for j in range(H)])
# OV copy test: T^T OV T
print("OV self-map test T^T OV T id-score:")
for l in range(4):
    print(" layer",l,[round(idscore(T.T@HS[l][j]["OV"]@T),3) for j in range(H)])
```