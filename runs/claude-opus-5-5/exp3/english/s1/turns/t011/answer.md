```python
import numpy as np
dims=[78,72,222,239,311,35,273,356,300,162,223,60,136,114,360]
print('dim  ln1w0  ln1b0  ln2w0  ln2b0 ln1w1 ln1b1 ln1w2 ln1b2 ln1w3 ln1b3 | rownorm Wo0 Wp0 Wo1 Wp1 | colnorm q0 k0 v0 fc0')
for d in dims:
    print(d, ' '.join(f"{W[n][d]:6.2f}" for n in ['ln1_w.0','ln1_b.0','ln2_w.0','ln2_b.0','ln1_w.1','ln1_b.1','ln1_w.2','ln1_b.2','ln1_w.3','ln1_b.3']),'|',
          ' '.join(f"{np.linalg.norm(W[n][d]):5.2f}" for n in ['W_o.0','W_proj.0','W_o.1','W_proj.1']),'|',
          ' '.join(f"{np.linalg.norm(W[n][:,d]):5.2f}" for n in ['W_q.0','W_k.0','W_v.0','W_fc.0']))
print('typical rownorms', [round(float(np.median(np.linalg.norm(W[n],axis=1))),2) for n in ['W_o.0','W_proj.0','W_o.1','W_proj.1']])
print('typical colnorms', [round(float(np.median(np.linalg.norm(W[n],axis=0))),2) for n in ['W_q.0','W_k.0','W_v.0','W_fc.0']])
# largest ln1_b.1 dims
for n in ['ln1_b.1','ln1_b.2','ln2_b.0']:
    i=np.argsort(-np.abs(W[n]))[:8]; print(n,i,np.round(W[n][i],2))
```