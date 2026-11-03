```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
rng=np.random.default_rng(1)
def eff(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]@np.diag(W[f'ln1_w.{l}'])
def OVe(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,s]@eff('W_v',l,j)
def ratio(R,A,n=2):
    num=np.linalg.norm(R@P@A); bs=[]
    for _ in range(n):
        Q,_=np.linalg.qr(rng.standard_normal((d,d))); bs.append(np.linalg.norm(R@P@Q@A))
    return num/np.mean(bs)
srcs=[(f"L{l}H{j}",OVe(l,j)) for l in range(3) for j in range(6)]+[(f"MLP{l}",W[f'W_proj.{l}']) for l in range(3)]
print("sources:", [s[0] for s in srcs])
for l in range(1,4):
    for j in range(6):
        for nm in ['W_k','W_q','W_v']:
            R=eff(nm,l,j)
            r=[ratio(R,A) for (n,A) in srcs if (n.startswith('L') and int(n[1])<l) or (n.startswith('MLP') and int(n[3])<l)]
            print(f"L{l}H{j} {nm[2]}:", np.array(r))
    # MLP reader
    R=W[f'W_fc.{l}']@np.diag(W[f'ln2_w.{l}'])
    r=[ratio(R,A) for (n,A) in srcs if (n.startswith('L') and int(n[1])<=l) or (n.startswith('MLP') and int(n[3])<l)]
    print(f"MLP{l} fc:", np.array(r), " (includes same-layer heads)")
```