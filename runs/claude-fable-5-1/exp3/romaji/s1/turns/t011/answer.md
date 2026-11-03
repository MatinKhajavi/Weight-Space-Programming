```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
OV={}
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    OV[j]=W['W_o.0'][:,s]@(W['W_v.0'][s]*ln0)
heads=[]
for l in [1,2,3]:
    ln=W[f'ln1_w.{l}']; lb=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Q=W[f'W_q.{l}'][s]; K=W[f'W_k.{l}'][s]
        cq=Q@lb                      # constant query component
        c=(K*ln).T@cq/8              # key-side preference vector (384)
        heads.append((l,j,c,Q,K))
print("norm of key-preference vector c, and its fraction on special dims, and projection through L0 heads' OV:")
ws={}
for (l,j,c,Q,K) in heads:
    sp=np.linalg.norm(c[[21,39,58,99,184,383]])**2/np.linalg.norm(c)**2
    w5=OV[5].T@c; w1=OV[1].T@c; w2=OV[2].T@c; w4=OV[4].T@c; w3=OV[3].T@c; w0=OV[0].T@c
    ws[(l,j)]=(w5,w1,w2,w4,w3,w0)
    print(f"L{l}H{j}: |c|={np.linalg.norm(c):.2f} frac special={sp:.2f}  |OV5^T c|={np.linalg.norm(w5):.2f} |OV1^T c|={np.linalg.norm(w1):.2f} |OV2^T c|={np.linalg.norm(w2):.2f} |OV4^T c|={np.linalg.norm(w4):.2f} |OV3^T c|={np.linalg.norm(w3):.2f} |OV0^T c|={np.linalg.norm(w0):.2f}   c on dims 21,39,58,99,184: {c[[21,39,58,99,184]]}")
print()
keys=list(ws.keys())
def cos(a,b): return a@b/np.linalg.norm(a)/np.linalg.norm(b)
print("cosine of w5 (dependence on L0H5 input) across heads:")
Mx=np.array([[cos(ws[a][0],ws[b][0]) for b in keys] for a in keys])
print("     "+" ".join(f"{l}{j}" for l,j in keys))
for i,a in enumerate(keys): print(f"L{a[0]}H{a[1]}", " ".join(f"{v:+.1f}" for v in Mx[i]))
print("cosine of w1 (dependence on L0H1 input) across heads:")
Mx=np.array([[cos(ws[a][1],ws[b][1]) for b in keys] for a in keys])
for i,a in enumerate(keys): print(f"L{a[0]}H{a[1]}", " ".join(f"{v:+.1f}" for v in Mx[i]))
```