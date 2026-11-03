```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for l in range(4):
    w1=W[f'ln1_w.{l}']
    print('layer',l,'ln1_w quantiles',np.quantile(w1,[0,.05,.25,.5,.75,.95,1]))
    print('  ln2_w quantiles',np.quantile(W[f'ln2_w.{l}'],[0,.05,.25,.5,.75,.95,1]))
    for h in range(6):
        s=slice(64*h,64*h+64)
        q=W[f'W_q.{l}'][s]*w1; k=W[f'W_k.{l}'][s]*w1; v=W[f'W_v.{l}'][s]*w1; o=W[f'W_o.{l}'][:,s]
        A=q.T@k
        sym=np.linalg.norm(A+A.T); asym=np.linalg.norm(A-A.T)
        OV=o@v
        ev=np.linalg.eigvals(OV)
        print(f'  H{h} |q|={np.linalg.norm(q):.2f} |k|={np.linalg.norm(k):.2f} |v|={np.linalg.norm(v):.2f} |o|={np.linalg.norm(o):.2f} |QK|={np.linalg.norm(A):.2f} asym/sym={asym/sym:.3f} trQK/|QK|={np.trace(A)/np.linalg.norm(A):.3f} OVtr/sum|ev|={ev.real.sum()/np.abs(ev).sum():.3f}')
```