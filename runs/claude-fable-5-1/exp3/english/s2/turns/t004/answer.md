```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]
s2=slice(128,192)
Wq2=W["W_q.0"][s2]*g; Wk2=W["W_k.0"][s2]*g   # effective on normalized x
QK=Wq2.T@Wk2
U,S,Vt=np.linalg.svd(QK)
print("L0H2 QK sv:", S[:70])
# query-side and key-side subspaces
r=50
Pq=U[:,:r]; Pk=Vt[:r].T
pa=np.linalg.svd(Pq.T@Pk,compute_uv=False)
print("principal cosines between query and key subspaces (r=50):", pa[:10], pa[-10:])
# energy of each head's matrices inside key subspace P (combined)
P,_=np.linalg.qr(np.concatenate([Pq,Pk],axis=1))
# take top-r of combined via svd
Uc,Sc,_=np.linalg.svd(np.concatenate([Pq,Pk],axis=1))
print("combined sv:", Sc[:10], Sc[45:55], Sc[95:100])
P=Uc[:,:60]
def frac(M): return (np.linalg.norm(M@P)**2)/(np.linalg.norm(M)**2)
print("dim of P:",P.shape[1], "random expect frac", 60/384)
for j in range(H):
    s=slice(j*hd,(j+1)*hd)
    print(f"L0H{j}: q {frac(W['W_q.0'][s]*g):.2f} k {frac(W['W_k.0'][s]*g):.2f} v {frac(W['W_v.0'][s]*g):.2f}   (raw, no ln scaling: q {frac(W['W_q.0'][s]):.2f} k {frac(W['W_k.0'][s]):.2f} v {frac(W['W_v.0'][s]):.2f})")
g2=W["ln2_w.0"]
print("MLP0 fc frac:", frac(W["W_fc.0"]*g2), "raw", frac(W["W_fc.0"]))
# output sides: W_o.0 columns per head: fraction of output in P
for j in range(H):
    s=slice(j*hd,(j+1)*hd)
    Wo=W["W_o.0"][:,s]
    print(f"L0H{j}: o-out frac in P {np.linalg.norm(P.T@Wo)**2/np.linalg.norm(Wo)**2:.2f}")
print("proj0 out frac in P", np.linalg.norm(P.T@W["W_proj.0"])**2/np.linalg.norm(W["W_proj.0"])**2)
# later layers reading from P (on raw residual - but LN differs; approximate with their ln1_w)
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]; gl2=W[f"ln2_w.{l}"]
    print(f"layer {l}: ", " ".join(f"H{j} q{frac(W[f'W_q.{l}'][j*hd:(j+1)*hd]*gl):.2f} k{frac(W[f'W_k.{l}'][j*hd:(j+1)*hd]*gl):.2f} v{frac(W[f'W_v.{l}'][j*hd:(j+1)*hd]*gl):.2f}" for j in range(H)), f" fc {frac(W[f'W_fc.{l}']*gl2):.2f}")
```