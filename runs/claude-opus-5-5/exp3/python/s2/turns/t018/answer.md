```python
import numpy as np
w0=W['ln1_w.0']; w2=W['ln2_w.0']
F=W['W_fc.0']*w2
n=np.linalg.norm(F,axis=1)
print(np.round(np.sort(n)[::-1][:30],2)); print(np.round(np.quantile(n,[0.01,0.1,0.5,0.9,0.99]),2))
Fr=W['W_fc.0']; nr=np.linalg.norm(Fr,axis=1); print('raw',np.round(np.sort(nr)[::-1][:20],2), np.round(np.quantile(nr,[0.1,0.5,0.9]),2))
Pj=W['W_proj.0']; npj=np.linalg.norm(Pj,axis=0); print('proj col norms',np.round(np.sort(npj)[::-1][:20],2),np.round(np.quantile(npj,[0.1,0.5,0.9]),2))
print('corr fc norm vs proj norm',np.corrcoef(nr,npj)[0,1])
# top singular direction of F
u,s,vt=np.linalg.svd(F,full_matrices=False)
print('top sv dir coords', np.argsort(-np.abs(vt[0]))[:10], np.round(np.sort(np.abs(vt[0]))[::-1][:10],2))
print('u0 stats', np.round(np.quantile(u[:,0],[0,0.01,0.5,0.99,1]),3), 'frac positive', (u[:,0]>0).mean())
print('ln2_b.0 top coords', np.argsort(-np.abs(W['ln2_b.0']))[:10], np.round(W['ln2_b.0'][np.argsort(-np.abs(W['ln2_b.0']))[:10]],2))
print('F @ ln2_b (neuron biases) quantiles', np.round(np.quantile(W['W_fc.0']@W['ln2_b.0'],[0,0.1,0.5,0.9,1]),2))
```