```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
Uf,Sf,Vf=np.linalg.svd((fc*g2)[strong],full_matrices=False); T=Vf[:30].T
def mats(l):
    gl=W[f"ln1_w.{l}"]; out=[]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv,Wo=W[f"W_q.{l}"][s]*gl,W[f"W_k.{l}"][s]*gl,W[f"W_v.{l}"][s]*gl,W[f"W_o.{l}"][:,s]
        out.append((Wq.T@Wk, Wo@Wv))
    return out
allm=[mats(l) for l in range(4)]
def idscore(M): n=M.shape[0]; return np.trace(M)/np.sqrt(n)/np.linalg.norm(M)
A=allm[0][2][1]@T   # prev-token image of token directions (384x30)
B=allm[0][3][1]@T   # L0H3 image
# orthonormalize A columns to keep the id test meaningful
Aq,_=np.linalg.qr(A); Bq,_=np.linalg.qr(B)
print("tests per head: [tok_i vs tok_j] [tok_i vs prev_j] [prev_i vs tok_j] [prev_i vs prev_j]   (prev via L0H2) ; then via L0H3: [tok_i vs h3_j] [h3_i vs h3_j]")
for l in range(1,4):
    for j in range(H):
        QK=allm[l][j][0]
        r=[idscore(T.T@QK@T), idscore(T.T@QK@Aq), idscore(Aq.T@QK@T), idscore(Aq.T@QK@Aq), idscore(T.T@QK@Bq), idscore(Bq.T@QK@Bq)]
        print(f"L{l}H{j}: "+" ".join(f"{x:6.2f}" for x in r))
# also check the linear proxy of MLP0: M0 = W_proj0 diag(g2) W_fc0 ; test whether MLP0 output on token directions is 'copy-like' or orthogonal
M0=W["W_proj.0"]@(W["W_fc.0"]*g2)
print("MLP0 linear proxy: id-score on T:", idscore(T.T@M0@T), " on prev-image:", idscore(Aq.T@M0@Aq))
# Where does MLP0 write? overlap of W_proj.0 column space with T and with Aq
Pj=W["W_proj.0"]
print("proj0 output energy in T:", np.linalg.norm(T.T@Pj)**2/np.linalg.norm(Pj)**2, " in prev-image:", np.linalg.norm(Aq.T@Pj)**2/np.linalg.norm(Pj)**2, " expected 30/384=",30/384)
```