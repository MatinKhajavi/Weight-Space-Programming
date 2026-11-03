```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=np.diag(W['ln1_w.0'])
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
OV5=P@W['W_o.0'][:,5*hd:6*hd]@blk('W_v',0,5)@w0
U,S,Vt=np.linalg.svd(OV5)
for l in range(1,4):
    b=W[f'ln1_b.{l}']; wl=np.diag(W[f'ln1_w.{l}'])
    print(f"layer {l}: b on channel dirs:", [round(float(U[:,r]@b),2) for r in range(3)])
    for j in range(6):
        qb=blk('W_q',l,j)@b; k=blk('W_k',l,j)@wl; kd=k.T@qb
        fr=np.linalg.norm(U[:,:4].T@kd)**2/np.linalg.norm(kd)**2
        # also: the symmetric 'same-channel' term: channel->channel block of QK
        QK=(blk('W_q',l,j)@wl).T@k
        cc=U[:,:2].T@QK@U[:,:2]
        print(f"  L{l}H{j}: |qb|={np.linalg.norm(qb):5.2f} const-query key pref in channel: frac={fr:.2f} loads dir0..3: {U[:,:4].T@kd}  | channel x channel QK block: {cc.ravel()}")
# Which heads read the channel in V (to erase)?  V-read fraction in channel and sign of OV on channel: u^T OV u for channel dirs
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        OV=W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@blk('W_v',l,j)@wl
        vals=[round(float(U[:,r]@OV@U[:,r]),2) for r in range(3)]
        print(f"L{l}H{j} OV diag on channel dirs: {vals}", end="; ")
    print()
```