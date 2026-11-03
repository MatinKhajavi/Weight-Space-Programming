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
A2=allm[0][2][1]@T; A2=A2/np.linalg.norm(A2,axis=0,keepdims=True)   # prev-token images, column-normalized (keeps index correspondence)
A3=allm[0][3][1]@T; A3=A3/np.linalg.norm(A3,axis=0,keepdims=True)
# linearized MLP0 image of tokens
M0=W["W_proj.0"]@(W["W_fc.0"]*g2); Am=M0@T; Am=Am/np.linalg.norm(Am,axis=0,keepdims=True)
print("columns: [tok_i,tok_j] [tok_i,prev_j] [prev_i,tok_j] [prev_i,prev_j] [tok_i,h3_j] [h3_i,tok_j] [h3_i,h3_j] [mlp_i,mlp_j] [mlp_i,prev_j] [tok_i,mlp_j]")
for l in range(1,4):
    for j in range(H):
        QK=allm[l][j][0]
        r=[idscore(X.T@QK@Y) for X,Y in [(T,T),(T,A2),(A2,T),(A2,A2),(T,A3),(A3,T),(A3,A3),(Am,Am),(Am,A2),(T,Am)]]
        print(f"L{l}H{j}: "+" ".join(f"{x:6.2f}" for x in r))
# also OV copy structure for L3 heads: does OV map token t to something aligned with MLP0 image / T?
print("OV tests [T->T] [T->mlpimg] [prev->T]:")
for l in range(1,4):
    for j in range(H):
        OVm=allm[l][j][1]
        print(f"L{l}H{j}: {idscore(T.T@OVm@T):6.2f} {idscore(Am.T@OVm@T):6.2f} {idscore(T.T@OVm@A2):6.2f}")
```