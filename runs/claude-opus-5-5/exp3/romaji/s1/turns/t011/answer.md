```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
b0=W['ln1_b.0']
for h in range(6):
    s=slice(64*h,64*h+64)
    q,k,v,o=headmats(0,h)
    A=q.T@k
    U,S,Vt=np.linalg.svd(A)
    qb=W['W_q.0'][s]@b0
    r=k.T@qb
    print(f'L0H{h} sv(A) {S[:5]}  cos(u1,v1)={U[:,0]@Vt[0]:.2f} cos(u2,v2)={U[:,1]@Vt[1]:.2f}  |qb|={np.linalg.norm(qb):.2f} |r|={np.linalg.norm(r):.2f} cos(r,v1)={r@Vt[0]/np.linalg.norm(r):.2f}')
```