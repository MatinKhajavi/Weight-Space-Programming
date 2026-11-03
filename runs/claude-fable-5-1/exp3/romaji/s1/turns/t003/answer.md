```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
rng=np.random.default_rng(0)
def heads(l):
    ln=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Q=W[f'W_q.{l}'][s]*ln; K=W[f'W_k.{l}'][s]*ln; V=W[f'W_v.{l}'][s]*ln
        QK=Q.T@K
        OV=W[f'W_o.{l}'][:,s]@V
        out.append((QK,OV))
    return out
H=[heads(l) for l in range(4)]
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
def randrot(M):
    # random orthogonal rotation on both sides
    q1,_=np.linalg.qr(rng.standard_normal((384,384)))
    q2,_=np.linalg.qr(rng.standard_normal((384,384)))
    return q1@M@q2
for la in range(4):
    for lb in range(la+1,4):
        print(f"=== layer {la} -> layer {lb}")
        for kind in ['K','Q','V']:
            print(" ",kind,"-composition (rows: src head in layer",la,", cols: dst head in layer",lb,")")
            for a in range(6):
                OVa=H[la][a][1]
                row=[];base=[]
                for b in range(6):
                    QKb,OVb=H[lb][b]
                    if kind=='K': M=QKb; val=comp(M,OVa); bl=np.mean([comp(M,randrot(OVa)) for _ in range(3)])
                    elif kind=='Q': M=QKb.T; val=comp(M,OVa); bl=np.mean([comp(M,randrot(OVa)) for _ in range(3)])
                    else: M=OVb; val=comp(M,OVa); bl=np.mean([comp(M,randrot(OVa)) for _ in range(3)])
                    row.append(val); base.append(bl)
                print(f"   H{a}: ", " ".join(f"{v:.3f}({v/b:.1f}x)" for v,b in zip(row,base)))
```