```python
np.set_printoptions(precision=2, suppress=True, linewidth=200)
hd=64
def mats(l):
    lw = W[f'ln1_w.{l}']
    QK=[];OV=[]
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        Q=W[f'W_q.{l}'][sl]*lw[None,:]; K=W[f'W_k.{l}'][sl]*lw[None,:]; V=W[f'W_v.{l}'][sl]*lw[None,:]; O=W[f'W_o.{l}'][:,sl]
        QK.append(Q.T@K); OV.append(O@V)
    return QK,OV
def cosmat(L):
    n=len(L); M=np.zeros((n,n))
    for a in range(n):
        for b in range(n):
            M[a,b]=np.sum(L[a]*L[b])/np.linalg.norm(L[a])/np.linalg.norm(L[b])
    return M
for l in range(4):
    QK,OV=mats(l)
    print(f"layer {l} QK cosine:\n", cosmat(QK))
    print(f"layer {l} OV cosine:\n", cosmat(OV))
    # also QK symmetric vs antisymmetric parts
    for j in range(6):
        M=QK[j]; S=(M+M.T)/2; A=(M-M.T)/2
        print(f"  L{l}H{j}: sym frac {np.linalg.norm(S)**2/np.linalg.norm(M)**2:.2f}  trace/fro {np.trace(M)/np.linalg.norm(M):.2f}")
```