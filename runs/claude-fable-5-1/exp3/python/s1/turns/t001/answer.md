```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
for l in range(4):
    for name in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        M=W[f'{name}.{l}']
        print(l,name,M.shape, 'fro=%.2f std=%.4f'%(np.linalg.norm(M), M.std()))
    for name in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        v=W[f'{name}.{l}']
        print(l,name,'mean=%.3f std=%.3f min=%.3f max=%.3f'%(v.mean(),v.std(),v.min(),v.max()))
print()
# per head norms and spectra
for l in range(4):
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]; Wv=W[f'W_v.{l}'][s]; Wo=W[f'W_o.{l}'][:,s]
        QK=Wq.T@Wk; OV=Wo@Wv
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        # symmetry of QK
        sym=np.linalg.norm(QK+QK.T)/2/np.linalg.norm(QK); asym=np.linalg.norm(QK-QK.T)/2/np.linalg.norm(QK)
        # trace of OV (copying positive?)
        print(f'L{l}H{j} |q|={np.linalg.norm(Wq):.2f} |k|={np.linalg.norm(Wk):.2f} |v|={np.linalg.norm(Wv):.2f} |o|={np.linalg.norm(Wo):.2f} QKsv={sq[:6]} eff_rank={ (sq.sum()**2/(sq**2).sum()):.1f} sym={sym:.2f} asym={asym:.2f} OVsv={so[:5]} OVeffr={(so.sum()**2/(so**2).sum()):.1f} trOV={np.trace(OV):.2f}')
```