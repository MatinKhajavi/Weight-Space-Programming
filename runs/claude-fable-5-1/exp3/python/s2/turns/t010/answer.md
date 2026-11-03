```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
hd=64
u,s,vt = np.linalg.svd(W['W_proj.3'],full_matrices=False)
U = u[:,:28]
print("fraction of W_proj.3 energy in U:", (s[:28]**2).sum()/(s**2).sum())
# also check: W_o.3 energy in U
def fracU(M): return np.linalg.norm(U.T@M)**2/np.linalg.norm(M)**2
print("baseline:", 28/384)
for l in range(4):
    print(f"W_proj.{l} in U: {fracU(W[f'W_proj.{l}']):.3f}   W_o.{l} in U: {fracU(W[f'W_o.{l}']):.3f}")
    lw = W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        O = W[f'W_o.{l}'][:,sl]; V = W[f'W_v.{l}'][sl]*lw[None,:]
        OV = O@V
        print(f"   L{l}H{j}: W_o cols in U {fracU(O):.3f}; OV output in U {fracU(OV):.3f}")
# column norms of W_proj.l per neuron: fraction in U for alive neurons
for l in range(4):
    B = W[f'W_proj.{l}']; bn=np.linalg.norm(B,axis=0); fu = np.linalg.norm(U.T@B,axis=0)**2/bn**2
    alive = bn>np.percentile(bn,60)
    print(f"layer {l}: neuron out-in-U fractions (alive): median {np.median(fu[alive]):.3f}, 90pct {np.percentile(fu[alive],90):.3f}; frac neurons >0.5: {(fu[alive]>0.5).mean():.3f}")
```