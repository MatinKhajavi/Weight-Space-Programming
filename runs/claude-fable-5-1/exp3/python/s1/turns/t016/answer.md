```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
Ssum=np.zeros((d,d))
for (l2,h2) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(0,5)]; S=(M+M.T)/2; Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
T=U[:,::-1][:,:22]; PT=T@T.T; PP=np.eye(d)-PT
rows=np.vstack([head(0,1)[0],head(0,1)[1],head(0,5)[0],head(0,5)[1]])@PP
u,s,vt=np.linalg.svd(rows,full_matrices=False); P0=vt[:40].T
# low-frequency directions: V-input of L0H4 (top 4), projected to P0 span and orthonormalized
q4,k4,v4,o4=head(0,4)
u,s,vt=np.linalg.svd(v4@PP,full_matrices=False)
L=vt[:4].T   # low-freq dirs in residual (complement of T)
print('L0H4 V on complement sv:',s[:8])
Lq,_=np.linalg.qr(L)
# high-frequency: top Q/K dirs of L0H5 orthogonal to L
q5,k5,v5,o5=head(0,5)
u,s,vt=np.linalg.svd(np.vstack([q5,k5])@PP,full_matrices=False)
Hs=vt[:30].T; Hs=Hs-Lq@(Lq.T@Hs); Hq,_=np.linalg.qr(Hs)
for l in range(1):
    for j in range(6):
        q,k,v,o=head(l,j); QK=PP@(q.T@k)@PP
        tot=np.linalg.norm(QK)**2
        ll=np.linalg.norm(Lq.T@QK@Lq)**2/tot; lh=np.linalg.norm(Lq.T@QK@Hq)**2/tot; hl=np.linalg.norm(Hq.T@QK@Lq)**2/tot; hh=np.linalg.norm(Hq.T@QK@Hq)**2/tot
        # Q and K row energy in L and H
        ql=np.linalg.norm(q@Lq)**2/np.linalg.norm(q)**2; qh=np.linalg.norm(q@Hq)**2/np.linalg.norm(q)**2
        kl=np.linalg.norm(k@Lq)**2/np.linalg.norm(k)**2; kh=np.linalg.norm(k@Hq)**2/np.linalg.norm(k)**2
        vl=np.linalg.norm(v@Lq)**2/np.linalg.norm(v)**2; vh=np.linalg.norm(v@Hq)**2/np.linalg.norm(v)**2
        print(f'L0H{j}: QK(nonT) energy frac LL {ll:.2f} LH {lh:.2f} HL {hl:.2f} HH {hh:.2f} | Q in L {ql:.2f} H {qh:.2f} | K in L {kl:.2f} H {kh:.2f} | V in L {vl:.2f} H {vh:.2f}  (baseline L {4/384:.3f} H {30/384:.3f})')
# LN weights: which residual dims are amplified at layer 0, and are T / L / H concentrated there?
w=W['ln1_w.0']; idx=np.argsort(-w)
print('ln1_w.0 top 20:',w[idx[:20]])
for nm,Bs in [('T',T),('L',Lq),('H',Hq)]:
    e=(Bs**2).sum(1)  # energy per residual dim
    print(nm,'energy in top-40 ln dims:',e[idx[:40]].sum()/e.sum(), ' corr(energy, ln_w):',np.corrcoef(e,w)[0,1])
```