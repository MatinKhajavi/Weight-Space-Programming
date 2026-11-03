```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]
mask=np.ones(384,bool); mask[sink]=False
OVs={}
for h in range(6):
    q,k,v,o=headmats(0,h); OVs[h]=o@v
U3=np.linalg.svd(OVs[3])[0][:,:20]
U0=np.linalg.svd(OVs[0])[0][:,:10]
# layer-0 input subspace estimate: top right singular vecs of stacked layer-0 weights (n-space)
w0=W['ln1_w.0']
X=np.vstack([W['W_q.0']*w0,W['W_k.0']*w0,W['W_v.0']*w0, W['W_fc.0']*W['ln2_w.0']])
E=np.linalg.svd(X)[2][:60].T
def frac(M,B):  # fraction of row-energy of M (rows in residual) in subspace B
    return np.linalg.norm(M@B)**2/np.linalg.norm(M)**2
print('expected random frac for 20-dim:',20/379,' 10-dim',10/379,' 60-dim',60/379)
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=headmats(l,h)
        q=q[:,mask]; k=k[:,mask]; v=v[:,mask]
        B3=U3[mask]; B0=U0[mask]; BE=np.linalg.qr(E[mask])[0]; B3=np.linalg.qr(B3)[0]; B0=np.linalg.qr(B0)[0]
        print(f'L{l}H{h}  inL0H3out: q {frac(q,B3):.3f} k {frac(k,B3):.3f} v {frac(v,B3):.3f} | inL0H0out: q {frac(q,B0):.3f} k {frac(k,B0):.3f} v {frac(v,B0):.3f} | inE60: q {frac(q,BE):.3f} k {frac(k,BE):.3f} v {frac(v,BE):.3f}')
```