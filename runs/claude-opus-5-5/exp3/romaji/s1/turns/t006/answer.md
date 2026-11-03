```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
for h in range(6):
    q,k,v,o=headmats(0,h)
    OV=o@v
    s=np.linalg.svd(OV,compute_uv=False)
    sq=np.linalg.svd(q,compute_uv=False); sk=np.linalg.svd(k,compute_uv=False)
    print(h,'OV sv',s[:8],' q sv',sq[:5],' k sv',sk[:5])
U={}
for h in range(6):
    q,k,v,o=headmats(0,h)
    u,s,vt=np.linalg.svd(o@v)
    U[h]=(u[:,:3],s[:3],vt[:3])
print('cosines between top-1 output dirs')
print(np.array([[abs(U[a][0][:,0]@U[b][0][:,0]) for b in range(6)] for a in range(6)]))
print('cosines between top-1 input (read) dirs of OV')
print(np.array([[abs(U[a][2][0]@U[b][2][0]) for b in range(6)] for a in range(6)]))
```