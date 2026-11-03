```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
def OV(l,j):
    Wq,Wk,Wv,Wo=headmats(l,j); return Wo@Wv
def pos_ratio(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
writers={'L0H2':OV(0,2),'L0H3':OV(0,3),'L0H0':OV(0,0),'L0H5':OV(0,5)}
for l in [1,2]:
    for j in range(6): writers[f'L{l}H{j}']=OV(l,j)
for l in [1,2,3]:
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j); A=Wq.T@Wk
        out=[]
        for n,O in writers.items():
            if int(n[1])>=l: continue
            M=A@O
            out.append(f'{n}:{pos_ratio(M):+.2f}({np.linalg.norm(M):.1f})')
        print(f'L{l}H{j} QK·OVk', ' '.join(out))
```