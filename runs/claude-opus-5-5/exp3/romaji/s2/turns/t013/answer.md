```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K/8,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
def sv(A): return np.linalg.svd(A)
lw0=W['ln1_w.0']
U,S,Vt=sv(H[0][3]['OV']); Tok=np.linalg.qr(Vt[:20].T/lw0[:,None])[0]; Prev=U[:,:20]
U,S,Vt=sv(H[0][0]['OV']); Loc=U[:,:12]
U,S,Vt=sv(H[0][3]['QK']); Pos=np.linalg.qr(np.hstack([U[:,:20],Vt[:20].T])/lw0[:,None])[0][:,:24]
M=np.hstack([sv(H[0][h]['OV'])[0][:,:2] for h in [1,2,4,5]]); Pout=sv(M)[0][:,:2]
subs={'tok':Tok,'pos':Pos,'prev':Prev,'loc':Loc,'Pout':Pout}
for l in [1]:
  for h in range(6):
    U,S,Vt=sv(H[l][h]['OV']); subs[f'L{l}H{h}']=U[:,:16]
for l in [2]:
  for h in range(6):
    U,S,Vt=sv(H[l][h]['OV']); subs[f'L{l}H{h}']=U[:,:16]
names=list(subs)
for l in [1,2,3]:
  for h in range(6):
    QK=H[l][h]['QK']; tot=np.linalg.norm(QK)
    print(f'L{l}H{h} total {tot:.2f}')
    for a in ['tok','pos','prev','loc','Pout']+[n for n in names if n.startswith('L') and int(n[1])<l]:
        row=[]
        for c in ['tok','pos','prev','loc','Pout']+[n for n in names if n.startswith('L') and int(n[1])<l]:
            row.append(np.linalg.norm(subs[a].T@QK@subs[c]))
        print('  Q',a.ljust(5),np.array(row).round(2))
```