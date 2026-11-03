```python
import numpy as np
w0=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    Q=W['W_q.0'][s]*w0; K=W['W_k.0'][s]*w0; V=W['W_v.0'][s]*w0; O=W['W_o.0'][:,s]
    QK=Q.T@K; OV=O@V
    sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
    ev=np.linalg.eigvals(QK); 
    print(j,'|Q|',np.linalg.norm(Q).round(2),'|K|',np.linalg.norm(K).round(2),'|V|',np.linalg.norm(V).round(2),'|O|',np.linalg.norm(O).round(2))
    print('  QK sv',np.round(sq[:12],2),' trace/fro',round(np.trace(QK)/np.linalg.norm(QK),3), 'sym frac', round(np.linalg.norm((QK+QK.T)/2)/np.linalg.norm(QK),3))
    print('  OV sv',np.round(so[:12],2), ' trace/fro',round(np.trace(OV)/np.linalg.norm(OV),3))
```