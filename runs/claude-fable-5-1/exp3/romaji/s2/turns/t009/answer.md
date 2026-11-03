```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
def eff(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]@np.diag(W[f'ln1_w.{l}'])
def OVe(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,s]@eff('W_v',l,j)
def cos(a,b): return (a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b)
srcs=[(f"L{l}H{j}",OVe(l,j)) for l in range(3) for j in range(6)]+[(f"MLP{l}",W[f'W_proj.{l}']) for l in range(3)]
print("cos(OV_B P A, A): does head B re-write (+) or erase (-) source A's output?")
print("sources:", [s[0] for s in srcs])
for l in range(1,4):
    for j in range(6):
        B=OVe(l,j)
        r=[cos(B@P@A,A) for (n,A) in srcs if (n.startswith('L') and int(n[1])<l) or (n.startswith('MLP') and int(n[3])<l)]
        print(f"L{l}H{j}:", np.array(r))
# eigenvalue sign stats of OV
print("\nOV eigenvalue stats: sum of positive real parts, sum of negative real parts, top-5 |ev| with sign")
for l in range(4):
    for j in range(6):
        ev=np.linalg.eigvals(OVe(l,j)); re=ev.real
        idx=np.argsort(-np.abs(ev))[:5]
        print(f"L{l}H{j}: pos={re[re>0].sum():.1f} neg={re[re<0].sum():.1f} | top: {ev[idx]}")
```