```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
rng=np.random.default_rng(0)
def comp(A,B,nrand=20):
    # A: (64,384) reads residual; B: (384,384) writes residual. score ||A B|| /(|A||B|), baseline with random rotations
    s=np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
    bs=[]
    for _ in range(nrand):
        R,_=np.linalg.qr(rng.standard_normal((d,d)))
        bs.append(np.linalg.norm(A@R@B)/(np.linalg.norm(A)*np.linalg.norm(B)))
    return s, np.mean(bs), np.std(bs)
for l2 in range(1,4):
    for h2 in range(H):
        q,k,v,o=head(l2,h2)
        for l1 in range(l2):
            lineQ=[];lineK=[];lineV=[]
            for h1 in range(H):
                B=OV[(l1,h1)]
                sq,bq,_=comp(q,B,5); sk,bk,_=comp(k,B,5); sv,bv,_=comp(v,B,5)
                lineQ.append(sq/bq); lineK.append(sk/bk); lineV.append(sv/bv)
            print(f'L{l2}H{h2} <- L{l1}: Q {np.array(lineQ)}  K {np.array(lineK)}  V {np.array(lineV)}')
```