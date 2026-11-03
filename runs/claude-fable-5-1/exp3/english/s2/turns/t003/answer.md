```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
rng=np.random.default_rng(0)
def heads(l):
    out=[]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv,Wo=W[f"W_q.{l}"][s],W[f"W_k.{l}"][s],W[f"W_v.{l}"][s],W[f"W_o.{l}"][:,s]
        out.append(dict(q=Wq,k=Wk,v=Wv,o=Wo,QK=Wq.T@Wk,OV=Wo@Wv))
    return out
HS=[heads(l) for l in range(4)]
def comp(A,B):  # ||A B||_F/(||A|| ||B||)
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
def baseline(A,B,n=5):
    vals=[]
    for _ in range(n):
        Q,_=np.linalg.qr(rng.standard_normal((d,d)))
        vals.append(comp(A,Q@B@Q.T))
    return np.mean(vals)
for la in range(3):
    for lb in range(la+1,4):
        print(f"=== source layer {la} -> dest layer {lb}")
        for kind in ["K","Q","V"]:
            M=np.zeros((H,H)); Bm=np.zeros((H,H))
            for i in range(H):
                for j in range(H):
                    src=HS[la][i]["OV"]; 
                    if kind=="K": A=HS[lb][j]["QK"]; M[i,j]=comp(A,src); Bm[i,j]=baseline(A,src,2)
                    elif kind=="Q": A=HS[lb][j]["QK"].T; M[i,j]=comp(A,src); Bm[i,j]=baseline(A,src,2)
                    else: A=HS[lb][j]["OV"]; M[i,j]=comp(A,src); Bm[i,j]=baseline(A,src,2)
            print(kind,"-comp (rows: src head, cols: dst head); ratio to baseline")
            print(np.round(M/Bm,2))
```