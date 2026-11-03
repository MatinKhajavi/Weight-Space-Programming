```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
Uf,Sf,Vf=np.linalg.svd((fc*g2)[strong],full_matrices=False); T=Vf[:30].T
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g
U,S,Vt=np.linalg.svd(Wq2.T@Wk2)
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
lo=np.where(g<0.05)[0]; hi=np.where(g>0.5)[0]
fce=(fc*g2)[strong]
print("MLP0 strong neurons energy frac on lo dims (ln1_w<0.05):", (np.linalg.norm(fce[:,lo])**2/np.linalg.norm(fce)**2), " expected", len(lo)/384, "; on hi dims:", np.linalg.norm(fce[:,hi])**2/np.linalg.norm(fce)**2, "expected", len(hi)/384)
# L0H2 OV on lo dims?
Wv2=W["W_v.0"][s2]*g
print("L0H2 v energy on lo dims:", np.linalg.norm(Wv2[:,lo])**2/np.linalg.norm(Wv2)**2, " on hi:", np.linalg.norm(Wv2[:,hi])**2/np.linalg.norm(Wv2)**2)
print("T energy on lo dims:", np.linalg.norm(T[lo])**2/30, " on hi dims:", np.linalg.norm(T[hi])**2/30)
# Key/Query token structure in layer1-3: singular values of Wk_eff @ T and Wq_eff @ T
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wk=W[f"W_k.{l}"][s]*gl; Wq=W[f"W_q.{l}"][s]*gl; Wv=W[f"W_v.{l}"][s]*gl
        sk=np.linalg.svd(Wk@T,compute_uv=False); sq=np.linalg.svd(Wq@T,compute_uv=False); sv=np.linalg.svd(Wv@T,compute_uv=False)
        print(f"L{l}H{j}: k@T sv {sk[:5]} | q@T sv {sq[:5]} | v@T sv {sv[:5]}")
# Also layer-0 symmetric/antisymmetric on P for H2 and H3
for j in [2,3]:
    s=slice(j*hd,(j+1)*hd); QK=(W["W_q.0"][s]*g).T@(W["W_k.0"][s]*g); A=P.T@QK@P
    ev=np.linalg.eigvalsh((A+A.T)/2)
    print(f"L0H{j} QK on P: sym norm {np.linalg.norm(A+A.T)/2:.2f} asym norm {np.linalg.norm(A-A.T)/2:.2f}; sym eigs top {ev[-5:][::-1]} bottom {ev[:5]}")
```