```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
hd=64
L0 = {}
lw0 = W['ln1_w.0']
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    for nm in 'qkv':
        L0[f'{j}{nm}'] = W[f'W_{nm}.0'][s]*lw0[None,:]
# positional subspace from H1 and H5 q,k
Pm = np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']])
u,s,vt = np.linalg.svd(Pm, full_matrices=False)
print("P stack singular values:", s[:40])
Tm = np.vstack([L0['1v'],L0['5v']])
u2,s2,vt2 = np.linalg.svd(Tm, full_matrices=False)
print("T stack singular values:", s2[:40])
# choose ranks
rP, rT = 24, 30
P = vt[:rP].T; T = vt2[:rT].T
print("overlap P,T:", np.linalg.norm(P.T@T)**2, "(random expectation ~", rP*rT/384, ")")
def frac(M, S):  # fraction of row-space energy (M: k x 384) inside subspace S (384 x r)
    return np.linalg.norm(M@S)**2/np.linalg.norm(M)**2
print("baseline fractions: P", rP/384, "T", rT/384)
for l in range(4):
    lw = W[f'ln1_w.{l}']
    print("layer",l)
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        out=[]
        for nm in 'qkv':
            M = W[f'W_{nm}.{l}'][s]*lw[None,:]
            out.append(f"{nm}: P {frac(M,P):.2f} T {frac(M,T):.2f}")
        O = W[f'W_o.{l}'][:,s].T
        out.append(f"o: P {frac(O,P):.2f} T {frac(O,T):.2f}")
        print(f"  H{j} "+" | ".join(out))
```