```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
l=0
ln=W['ln1_w.0']; lb=W['ln1_b.0']
info={}
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    Q=W[f'W_q.{l}'][s]*ln; K=W[f'W_k.{l}'][s]*ln; V=W[f'W_v.{l}'][s]*ln
    M=Q.T@K
    U,S,Vt=np.linalg.svd(M)
    OV=W[f'W_o.{l}'][:,s]@V
    Uo,So,Vto=np.linalg.svd(OV)
    info[j]=dict(qdir=U[:,0],kdir=Vt[0],qdir2=U[:,1],kdir2=Vt[1],ovin=Vto[0],ovout=Uo[:,0],ovin2=Vto[1],ovout2=Uo[:,1],S=S[:4],So=So[:4])
    print(f"L0H{j}: QK S={S[:4]}  q-dir top dims {np.argsort(-np.abs(U[:,0]))[:8]}  k-dir top dims {np.argsort(-np.abs(Vt[0]))[:8]}")
    print(f"      OV S={So[:4]} in-dir top dims {np.argsort(-np.abs(Vto[0]))[:8]} out-dir top dims {np.argsort(-np.abs(Uo[:,0]))[:8]} vals {Uo[np.argsort(-np.abs(Uo[:,0]))[:5],0]}")
print()
print("cosines between heads' q-dir / k-dir / ov-in / ov-out (top singular):")
for a in range(6):
    for b in range(a+1,6):
        print(f" H{a}-H{b}: q {cos(info[a]['qdir'],info[b]['qdir']):+.2f} k {cos(info[a]['kdir'],info[b]['kdir']):+.2f} ovin {cos(info[a]['ovin'],info[b]['ovin']):+.2f} ovout {cos(info[a]['ovout'],info[b]['ovout']):+.2f}")
# relation between q-dir / k-dir and ov-in within head
for j in range(6):
    print(f" H{j}: cos(qdir,ovin)={cos(info[j]['qdir'],info[j]['ovin']):+.2f} cos(kdir,ovin)={cos(info[j]['kdir'],info[j]['ovin']):+.2f} cos(qdir,kdir)={cos(info[j]['qdir'],info[j]['kdir']):+.2f}")
```