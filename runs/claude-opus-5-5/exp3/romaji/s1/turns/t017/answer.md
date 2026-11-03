```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
U,S,Vt=np.linalg.svd(OV3)
kk=20
T=Vt[:kk].T            # input dirs (n-space at layer0)
P=OV3@T                # moved representation
P[sink]=0
f=lambda e: e.real.sum()/np.abs(e).sum()
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=headmats(l,h)
        A=q.T@k
        A[sink,:]=0; A[:,sink]=0
        blocks={'TT':T.T@A@T,'TP':T.T@A@P,'PP':P.T@A@P,'PT':P.T@A@T}
        s=' '.join(f'{n}:|{np.linalg.norm(B):.2f}| {f(np.linalg.eigvals(B)):+.2f} tr{np.trace(B):+.2f}' for n,B in blocks.items())
        print(f'L{l}H{h} {s}')
```