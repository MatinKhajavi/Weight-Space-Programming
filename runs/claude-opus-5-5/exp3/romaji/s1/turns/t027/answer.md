```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
Uo,So,Vto=np.linalg.svd(OV3)
T=Vto[:28].T
X=np.vstack([q3,k3]); Pz=np.linalg.svd(X)[2][:40].T
Pz=Pz-T@(T.T@Pz); Pz,_=np.linalg.qr(Pz)
w0=W['ln1_w.0']
Xall=np.vstack([W['W_q.0']*w0,W['W_k.0']*w0,W['W_v.0']*w0])
for h in [0,1,2,4,5]:
    q,k,v,o=headmats(0,h)
    for nm,M in [('q',q),('k',k),('v',v)]:
        U,S,Vt=np.linalg.svd(M)
        d=Vt[:3]
        print(f'L0H{h} {nm}: sv {S[:3]}  inT {np.round(np.linalg.norm(T.T@d.T,axis=0)**2,2)} inPos40 {np.round(np.linalg.norm(Pz.T@d.T,axis=0)**2,2)}')
```