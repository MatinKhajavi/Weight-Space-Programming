```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),24)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0]),30)
tw=(T**2).sum(1); pw=(P**2).sum(1)
order=np.argsort(-w0)
for name,arr in [('T',tw),('P',pw)]:
    print(name,'mass in top-50 lnw coords',arr[order[:50]].sum().round(2),'top-100',arr[order[:100]].sum().round(2),'bottom-200',arr[order[-200:]].sum().round(2), 'total',arr.sum().round(1))
print('corr lnw vs T mass',np.corrcoef(w0,tw)[0,1].round(2),' vs P mass',np.corrcoef(w0,pw)[0,1].round(2))
# raw (unfolded) versions
Tr,_=rs(np.vstack([W['W_v.0'][64:128], W['W_v.0'][320:384]]),24)
Pr,_=rs(np.vstack([W['W_q.0'][64:128], W['W_q.0'][320:384], W['W_k.0'][64:128], W['W_k.0'][320:384], W['W_v.0'][256:320]]),30)
print('raw corr lnw vs T mass',np.corrcoef(w0,(Tr**2).sum(1))[0,1].round(2),' vs P',np.corrcoef(w0,(Pr**2).sum(1))[0,1].round(2))
s=np.linalg.svd(np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])*w0,compute_uv=False); print(np.round(s[:80],2))
```