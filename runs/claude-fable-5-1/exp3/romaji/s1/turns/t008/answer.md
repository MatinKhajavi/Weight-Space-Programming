```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
rng=np.random.default_rng(1)
bias_dims=[21,39,58,99,184,383,163,153]
def heads(l, drop=False):
    ln=W[f'ln1_w.{l}'].copy()
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Q=W[f'W_q.{l}'][s]*ln; K=W[f'W_k.{l}'][s]*ln; V=W[f'W_v.{l}'][s]*ln
        if drop:
            K=K.copy(); K[:,bias_dims]=0
            Q=Q.copy(); Q[:,bias_dims]=0
        QK=Q.T@K
        OV=W[f'W_o.{l}'][:,s]@V
        out.append((QK,OV))
    return out
H=[heads(l,drop=True) for l in range(4)]
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
def randrot(M):
    q1,_=np.linalg.qr(rng.standard_normal((384,384))); q2,_=np.linalg.qr(rng.standard_normal((384,384)))
    return q1@M@q2
for la in range(3):
    for lb in range(la+1,4):
        print(f"=== layer {la} -> layer {lb} (bias dims dropped from Q,K)")
        for kind in ['K','Q']:
            print(" ",kind)
            for a in range(6):
                OVa=H[la][a][1]
                row=[]
                for b in range(6):
                    QKb=H[lb][b][0]
                    M=QKb if kind=='K' else QKb.T
                    val=comp(M,OVa); bl=np.mean([comp(M,randrot(OVa)) for _ in range(3)])
                    row.append((val,bl))
                print(f"   H{a}: "," ".join(f"{v:.3f}({v/b:.1f}x)" for v,b in row))
```