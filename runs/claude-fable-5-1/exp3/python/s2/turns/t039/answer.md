```python
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
lw0=W['ln1_w.0']; L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']])
u,s,vt=np.linalg.svd(Pm,full_matrices=False); Pd=vt[:24]   # P directions
ramp=Pd[:2].T; rel=Pd[2:].T
for h in range(6):
    M=L0[f'{h}q'].T@L0[f'{h}k']/8.0
    tot=np.linalg.norm(M)**2
    RR=ramp.T@M@ramp; QrelKrel=rel.T@M@rel; QrampKrel=ramp.T@M@rel; QrelKramp=rel.T@M@ramp
    # also constant-ish component: use ln1_b.0 direction? skip
    print(f"L0H{h}: ramp-ramp block {RR.round(4).tolist()} energy frac {np.linalg.norm(RR)**2/tot:.3f} | rel-rel frac {np.linalg.norm(QrelKrel)**2/tot:.3f} | ramp(q)-rel(k) {np.linalg.norm(QrampKrel)**2/tot:.3f} | rel(q)-ramp(k) {np.linalg.norm(QrelKramp)**2/tot:.3f} | total P-P frac {np.linalg.norm(Pd@M@Pd.T)**2/tot:.3f}")
    sv=np.linalg.svd(M,compute_uv=False); print("    top sv*384:", (sv[:5]*384).round(1))
# singular values of ramp directions in V of H0,H4,H2,H3 (what they average)
for h in [0,4,2,3]:
    V=L0[f'{h}v']; print(f"L0H{h} V energy on ramp dirs: {np.linalg.norm(V@ramp)**2/np.linalg.norm(V)**2:.2f}, on rel dirs: {np.linalg.norm(V@rel)**2/np.linalg.norm(V)**2:.2f}")
```