```python
import numpy as np
for d in [222,78,360,72,254,25,197,103,273]:
    print(d, [round(float(np.linalg.norm(W['W_o.0'][d,64*h:64*h+64])),2) for h in range(6)],
          'Wo1',[round(float(np.linalg.norm(W['W_o.1'][d,64*h:64*h+64])),2) for h in range(6)],
          'Wo2',[round(float(np.linalg.norm(W['W_o.2'][d,64*h:64*h+64])),2) for h in range(6)])
# For each L0 head: OV output, which dims dominate
for h in range(6):
    s=slice(64*h,64*h+64)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*W['ln1_w.0'])
    rn=np.linalg.norm(OV,axis=1); i=np.argsort(-rn)[:8]
    bias_out=W['W_o.0'][:,s]@(W['W_v.0'][s]@W['ln1_b.0'])
    j=np.argsort(-np.abs(bias_out))[:6]
    print('L0H',h,'OV row norms top',i,np.round(rn[i],2),' total',round(float(np.linalg.norm(OV)),2),'| bias out top',j,np.round(bias_out[j],2))
```