```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
L0={}
lw0=W['ln1_w.0']
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv':
        L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']])
u,s,vt=np.linalg.svd(Pm,full_matrices=False)
Pdirs=vt[:30]   # 30 x 384
print("energy of each layer-0 matrix on the top 30 P directions (fraction of its total energy):")
for j in range(6):
    for nm in 'qkv':
        M=L0[f'{j}{nm}']
        e=np.linalg.norm(M@Pdirs.T,axis=0)**2/np.linalg.norm(M)**2
        print(f"L0H{j}{nm}: "+" ".join(f"{x:.2f}" for x in e[:16]), f" total30={e.sum():.2f}")
# also ln2 / MLP0 reading of these directions and layer1 q/k
for l in range(1,3):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        for nm in 'qk':
            M=W[f'W_{nm}.{l}'][sl]*lw[None,:]
            e=np.linalg.norm(M@Pdirs.T,axis=0)**2/np.linalg.norm(M)**2
            print(f"L{l}H{j}{nm}: "+" ".join(f"{x:.2f}" for x in e[:16]), f" total30={e.sum():.2f}")
A=W['W_fc.0']*W['ln2_w.0'][None,:]
e=np.linalg.norm(A@Pdirs.T,axis=0)**2/np.linalg.norm(A)**2
print("W_fc.0 energy on P dirs:", e[:16], "total", e.sum())
```