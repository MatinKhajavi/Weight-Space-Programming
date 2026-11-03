```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
for h in [3,0,5]:
    q,k,v,o=headmats(0,h)
    print('L0H',h)
    print(' q ',np.linalg.svd(q,compute_uv=False)[:64])
    print(' k ',np.linalg.svd(k,compute_uv=False)[:64])
    print(' v ',np.linalg.svd(v,compute_uv=False)[:64])
    print(' OV',np.linalg.svd(o@v,compute_uv=False)[:64])
# whole-layer input-space spectrum
w0=W['ln1_w.0']
X=np.vstack([W['W_q.0']*w0,W['W_k.0']*w0,W['W_v.0']*w0])
print('qkv stacked', np.linalg.svd(X,compute_uv=False)[::4])
w2=W['ln2_w.0']
print('fc0 ', np.linalg.svd(W['W_fc.0']*w2,compute_uv=False)[::4])
print('raw W_q.0 (no ln)', np.linalg.svd(np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']]),compute_uv=False)[::4])
```