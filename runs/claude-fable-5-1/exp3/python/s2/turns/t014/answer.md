```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
lw0=W['ln1_w.0']
L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
outs=[]
for h in range(6):
    sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']
    u,s,vt=np.linalg.svd(OV,full_matrices=False); outs.append(u[:,:(27 if h in (1,2,3,5) else 3)])
# build an orthonormal basis of the union of all these subspaces for a cleaner decomposition
names=['P','T']+[f'L0H{h}out' for h in range(6)]
subs=[P,T]+outs
def decomp(M):
    # sequential Gram-Schmidt projection: energy explained by union, then per-subspace (non-orthogonal) fractions
    res={}
    tot=np.linalg.norm(M)**2
    for n,S in zip(names,subs): res[n]=np.linalg.norm(M@S)**2/tot
    allS=np.hstack(subs); Q,_=np.linalg.qr(allS)
    res['union']=np.linalg.norm(M@Q)**2/tot; res['uniondim']=Q.shape[1]
    return res
for l in range(4):
    A=W[f'W_fc.{l}']*W[f'ln2_w.{l}'][None,:]
    an=np.linalg.norm(A,axis=1); alive=A[an>0.8]
    print(f"W_fc.{l}: alive {alive.shape[0]} of 1536;", {k:(round(v,3) if isinstance(v,float) else v) for k,v in decomp(alive).items()})
for l in range(1,4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        print(f"L{l}H{j}: "+" | ".join(nm+" "+" ".join(f"{k[:3]}={v:.2f}" for k,v in decomp(W[f'W_{nm}.{l}'][sl]*lw[None,:]).items() if k!='uniondim') for nm in 'qkv'))
# union dim baseline
print("baseline union frac:", (24+30+27*4+3*2)/384)
```