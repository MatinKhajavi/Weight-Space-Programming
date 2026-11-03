```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for l in range(4):
    for nm in ['ln1_w','ln2_w']:
        w=W[f'{nm}.{l}']
        idx=np.argsort(w)
        print(f'{nm}.{l} smallest dims',idx[:5],w[idx[:5]],' largest',idx[-5:],w[idx[-5:]])
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h)
    u,s,vt=np.linalg.svd(o@v)
    for i in range(2):
        top=np.argsort(-np.abs(u[:,i]))[:5]
        print(f'L0H{h} out dir {i} (sv {s[i]:.2f}) top coords',top,u[top,i], ' read dir top coords', np.argsort(-np.abs(vt[i]))[:5], vt[i][np.argsort(-np.abs(vt[i]))[:5]])
```