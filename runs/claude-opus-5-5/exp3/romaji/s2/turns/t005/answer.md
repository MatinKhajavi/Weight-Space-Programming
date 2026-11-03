```python
import numpy as np
l=0; b=W['ln1_b.0']
for h in [0,1,2,3,4,5]:
    d=H[0][h]
    U,S,Vt=np.linalg.svd(d['QK']/8)
    s=slice(64*h,64*h+64)
    qc=W['W_q.0'][s]@b; kc=W['W_k.0'][s]@b
    keypref=d['K'].T@qc/8   # in normalized-x space
    qpref=d['Q'].T@kc/8
    print(h,'|qc|=%.3f |kc|=%.3f |keypref|=%.3f |qpref|=%.3f'%(np.linalg.norm(qc),np.linalg.norm(kc),np.linalg.norm(keypref),np.linalg.norm(qpref)),
          'cos(keypref,v1)=%.3f'%(keypref@Vt[0]/np.linalg.norm(keypref)), 'qc.kc/8=%.3f'%(qc@kc/8))
# overlap of OV output subspaces (top-3 left singular vectors)
outs={}
for h in range(6):
    U,S,Vt=np.linalg.svd(H[0][h]['OV']); outs[h]=(U[:,:3],Vt[:3])
for a in range(6):
    print(a,[np.round(np.linalg.svd(outs[a][0].T@outs[c][0],compute_uv=False),2) for c in range(6)])
print('input sides')
for a in range(6):
    print(a,[np.round(np.linalg.svd(outs[a][1]@outs[c][1].T,compute_uv=False),2) for c in range(6)])
```