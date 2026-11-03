```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
hd=64
lw0=W['ln1_w.0']
def OV0(h):
    sl=slice(h*hd,(h+1)*hd); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*lw0[None,:])
outs={}
for h in range(6):
    u,s,vt=np.linalg.svd(OV0(h),full_matrices=False)
    r = 27 if h in (1,2,3,5) else 3
    outs[h]=u[:,:r]
    print(f"L0H{h} OV sv:", s[:30])
def fr(M,S): return np.linalg.norm(M@S)**2/np.linalg.norm(M)**2
print("baselines:", {h:outs[h].shape[1]/384 for h in outs})
# who reads each L0 head's output subspace
for l in range(1,4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        row=[]
        for nm in 'qkv':
            M=W[f'W_{nm}.{l}'][sl]*lw[None,:]
            row.append(nm+":"+" ".join(f"{fr(M,outs[h]):.2f}" for h in range(6)))
        print(f"L{l}H{j} "+" | ".join(row))
for l in range(0,4):
    A=W[f'W_fc.{l}']*W[f'ln2_w.{l}'][None,:]
    an=np.linalg.norm(A,axis=1); alive=an>np.percentile(an,50)
    print(f"W_fc.{l} (alive neurons) mean frac in L0 head outs:", [f"{np.mean([fr(A[i:i+1],outs[h]) for i in np.where(alive)[0]]):.3f}" for h in range(6)])
    print(f"W_fc.{l} (alive) frac neurons with >0.4 in L0H1/H5 outs:", [f"{np.mean([fr(A[i:i+1],outs[h])>0.4 for i in np.where(alive)[0]]):.3f}" for h in range(6)])
```