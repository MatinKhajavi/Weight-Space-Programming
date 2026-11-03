```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
rng=np.random.default_rng(0)
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g
U,S,Vt=np.linalg.svd(Wq2.T@Wk2)
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
Uf,Sf,Vf=np.linalg.svd((fc*g2)[strong],full_matrices=False); T=Vf[:30].T
def frac(M,Q): return np.linalg.norm(M@Q)**2/np.linalg.norm(M)**2
def effrank(s): p=s**2/np.sum(s**2); return np.exp(-(p*np.log(p+1e-12)).sum())
for l in range(4):
    gl=W[f"ln1_w.{l}"]
    print(f"=== layer {l}")
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv=W[f"W_q.{l}"][s]*gl,W[f"W_k.{l}"][s]*gl,W[f"W_v.{l}"][s]*gl
        QK=Wq.T@Wk; sv=np.linalg.svd(QK,compute_uv=False)
        # init baseline for this layer's QK scale
        Rq=rng.standard_normal((hd,d))*0.02*gl; Rk=rng.standard_normal((hd,d))*0.02*gl; svb=np.linalg.svd(Rq.T@Rk,compute_uv=False)
        print(f"H{j}: QK sv {sv[:5]} (init top {svb[0]:.2f}) effrank {effrank(sv):.1f} | q:T {frac(Wq,T):.2f} P {frac(Wq,P):.2f} | k:T {frac(Wk,T):.2f} P {frac(Wk,P):.2f} | v:T {frac(Wv,T):.2f} P {frac(Wv,P):.2f} |q|{np.linalg.norm(Wq):.1f} |k|{np.linalg.norm(Wk):.1f}")
```