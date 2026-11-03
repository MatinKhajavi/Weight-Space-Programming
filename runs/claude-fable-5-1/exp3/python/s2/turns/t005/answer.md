```python
np.set_printoptions(precision=2, suppress=True, linewidth=200)
hd=64
def effrank(s):
    p = s**2/np.sum(s**2); return 1/np.sum(p**2)
for l in range(4):
    lw = W[f'ln1_w.{l}']
    print("=== layer", l)
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Q = W[f'W_q.{l}'][s]*lw[None,:]; K = W[f'W_k.{l}'][s]*lw[None,:]; V = W[f'W_v.{l}'][s]*lw[None,:]; O=W[f'W_o.{l}'][:,s]
        QK = Q.T@K/np.sqrt(hd); OV = O@V
        sq = np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        print(f"H{j} QK sv: {sq[:8]} effrank {effrank(sq):.1f} sum {sq.sum():.1f} | OV sv: {so[:6]} effrank {effrank(so):.1f}")
```