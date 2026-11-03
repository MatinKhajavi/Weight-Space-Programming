```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
Uo,So,Vto=np.linalg.svd(OV3)
T=Vto[:24].T
P=OV3@T
tops=[]
for l in [1,2]:
    for h in range(6):
        q,k,v,o=headmats(l,h)
        A=q.T@k
        B=T.T@A@T
        ev=np.linalg.eigvals(B); ev=ev[np.argsort(ev.real)]
        Bs=(B+B.T)/2; es=np.linalg.eigvalsh(Bs)
        print(f'L{l}H{h} TT eig (sorted by real):',np.round(ev[:8],2),'... max',np.round(ev[-3:],2))
        tops.append(np.linalg.eigh(Bs)[1][:,0])
tops=np.array(tops)
print('cos between most-negative sym eigvecs of TT across L1/L2 heads')
print(np.round(np.abs(tops@tops.T),2))
```