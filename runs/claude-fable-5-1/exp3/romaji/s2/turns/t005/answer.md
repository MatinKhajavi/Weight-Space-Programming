```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
def eff(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]@np.diag(W[f'ln1_w.{l}'])
def OVe(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,s]@eff('W_v',l,j)
rng=np.random.default_rng(0)
def randrot(A):
    Q,_=np.linalg.qr(rng.standard_normal((d,d))); return Q@A@Q.T
OV0=[OVe(0,a) for a in range(6)]
MLP0=W['W_proj.0']
print("K-composition (ratio to random-rotation baseline), rows=head B, cols=L0H0..5, MLP0")
for l in range(1,4):
    for j in range(6):
        Wk=eff('W_k',l,j); Wq=eff('W_q',l,j); Wv=eff('W_v',l,j)
        row=[]
        for A in OV0:
            num=np.linalg.norm(Wk@P@A); base=np.mean([np.linalg.norm(Wk@P@randrot(A)) for _ in range(3)]); row.append(num/base)
        num=np.linalg.norm(Wk@P@MLP0); base=np.mean([np.linalg.norm(Wk@P@randrot(MLP0@MLP0.T)@MLP0) for _ in range(1)]); 
        # simpler baseline for MLP: random rotation of rows
        Qr,_=np.linalg.qr(rng.standard_normal((d,d))); base=np.linalg.norm(Wk@P@Qr@MLP0); row.append(num/base)
        rowq=[]
        for A in OV0:
            num=np.linalg.norm(Wq@P@A); base=np.mean([np.linalg.norm(Wq@P@randrot(A)) for _ in range(3)]); rowq.append(num/base)
        Qr,_=np.linalg.qr(rng.standard_normal((d,d))); rowq.append(np.linalg.norm(Wq@P@MLP0)/np.linalg.norm(Wq@P@Qr@MLP0))
        rowv=[]
        for A in OV0:
            num=np.linalg.norm(Wv@P@A); base=np.mean([np.linalg.norm(Wv@P@randrot(A)) for _ in range(3)]); rowv.append(num/base)
        Qr,_=np.linalg.qr(rng.standard_normal((d,d))); rowv.append(np.linalg.norm(Wv@P@MLP0)/np.linalg.norm(Wv@P@Qr@MLP0))
        print(f"L{l}H{j} K:",np.array(row)," Q:",np.array(rowq)," V:",np.array(rowv))
```