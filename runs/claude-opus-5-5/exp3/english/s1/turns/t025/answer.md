```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
V2=W['W_v.0'][s2]*w0
OV2=W['W_o.0'][:,s2]@V2
w3=W['ln1_w.3']; s=slice(64,128)
B=(W['W_q.3'][s]*w3).T@(W['W_k.3'][s]*w3)
OV31=W['W_o.3'][:,s]@(W['W_v.3'][s]*w3)
print('OV2 sv',np.round(np.linalg.svd(OV2,compute_uv=False)[:40],2))
print('OV31 sv',np.round(np.linalg.svd(OV31,compute_uv=False)[:45],2))
M=B@OV2
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))]
print('ind composite eig', np.round(ev[:40],2))
print('ind composite sv', np.round(np.linalg.svd(M,compute_uv=False)[:40],2))
U3,S3,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
print('OUT-projected OV31 sv', np.round(np.linalg.svd(U3[:,:30].T@OV31,compute_uv=False)[:30],2))
```