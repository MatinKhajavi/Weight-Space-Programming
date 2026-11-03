```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
U,S,Vt=np.linalg.svd(OV3)
for kk in [18,24,28]:
    T=Vt[:kk].T
    print('kk',kk)
    for h in range(6):
        q,k,v,o=headmats(0,h)
        A=q.T@k
        Uq,Sa,Vk=np.linalg.svd(A)
        # fraction of top query/key directions in T
        fq=[np.linalg.norm(T.T@Uq[:,i])**2 for i in range(3)]
        fk=[np.linalg.norm(T.T@Vk[i])**2 for i in range(3)]
        Uv,Sv,Vv=np.linalg.svd(o@v)
        fv=[np.linalg.norm(T.T@Vv[i])**2 for i in range(3)]
        PT=np.eye(384)-T@T.T
        print(f'  L0H{h}: QK in T-T {np.linalg.norm(T.T@A@T)/np.linalg.norm(A):.2f}  Tc-Tc {np.linalg.norm(PT@A@PT)/np.linalg.norm(A):.2f}  T-Tc {np.linalg.norm(T.T@A@PT)/np.linalg.norm(A):.2f} Tc-T {np.linalg.norm(PT@A@T)/np.linalg.norm(A):.2f} | top q dirs in T {np.round(fq,2)} top k dirs in T {np.round(fk,2)} top v dirs in T {np.round(fv,2)}')
```