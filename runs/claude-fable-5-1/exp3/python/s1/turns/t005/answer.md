```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
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
    M=(q.T@k)@OV[(0,5)]
    S=(M+M.T)/2
    Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
print('combined eig top 40:',ev[::-1][:40])
print('combined eig bottom 10:',ev[:10])
for kT in [20,24,27]:
    T=U[:,::-1][:,:kT]
    PT=T@T.T
    print(f'--- token subspace dim {kT}; energy fractions in T (baseline {kT/384:.3f})')
    for l in range(4):
        for j in range(H):
            q,k,v,o=head(l,j)
            fq=np.linalg.norm(q@PT)**2/np.linalg.norm(q)**2
            fk=np.linalg.norm(k@PT)**2/np.linalg.norm(k)**2
            fv=np.linalg.norm(v@PT)**2/np.linalg.norm(v)**2
            fo=np.linalg.norm(PT@o)**2/np.linalg.norm(o)**2
            QK=q.T@k
            tt=np.linalg.norm(PT@QK@PT); pp=np.linalg.norm((np.eye(d)-PT)@QK@(np.eye(d)-PT)); tp=np.linalg.norm(PT@QK@(np.eye(d)-PT)); pt=np.linalg.norm((np.eye(d)-PT)@QK@PT)
            print(f'L{l}H{j} Q:{fq:.2f} K:{fk:.2f} V:{fv:.2f} O:{fo:.2f} | QK blocks TT:{tt:.1f} TP:{tp:.1f} PT:{pt:.1f} PP:{pp:.1f}  tr(QK on T)={np.trace(PT@QK@PT):.2f}')
```