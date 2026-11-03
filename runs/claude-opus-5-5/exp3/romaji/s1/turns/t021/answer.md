```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
qb,kb,vb,ob=headmats(3,3)
ev=np.linalg.eigvals(kb@OV3@qb.T)
ev=ev[np.argsort(-ev.real)]
print(np.round(ev,3))
M=qb.T@kb@OV3   # 384x384 bilinear: query n_i (current), key's source n (prev token)
# symmetric part eigen
S=(M+M.T)/2
es=np.linalg.eigvalsh(S)
print('sym part eig top', np.round(es[::-1][:40],3))
print('sym part eig bottom', np.round(es[:20],3))
```