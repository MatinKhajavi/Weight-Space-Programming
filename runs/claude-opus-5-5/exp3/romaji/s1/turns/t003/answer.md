```python
for l in range(2):
    w1=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        q=W[f'W_q.{l}'][s]*w1; k=W[f'W_k.{l}'][s]*w1
        # eigenvalues of k q^T (64x64) equal nonzero eigs of q^T k
        ev=np.linalg.eigvals(k@q.T)
        idx=np.argsort(-np.abs(ev))[:12]
        print(l,h,' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx]))
```