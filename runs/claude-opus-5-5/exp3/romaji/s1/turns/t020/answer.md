```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
for l,h in [(3,3),(3,0),(3,4),(2,0),(1,0)]:
    q,k,v,o=headmats(l,h)
    print(f'L{l}H{h} OV sv', np.linalg.svd(o@v,compute_uv=False)[:40])
    print(f'      QK sv', np.linalg.svd(q.T@k,compute_uv=False)[:40])
```