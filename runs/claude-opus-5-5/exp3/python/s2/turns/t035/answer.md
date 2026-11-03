```python
import numpy as np
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),40)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0,W['W_v.0'][0:64]*w0]),24)
alive=np.linalg.norm(W['W_fc.0'],axis=1)>1
F=(W['W_fc.0']*w2)[alive]
# token subspace: directions read by MLP0 within T, after removing P
Tp=T-P@(P.T@T); Tp,_s,_=np.linalg.svd(Tp,full_matrices=False); 
print('Tp sv (how much of T survives removing P)',np.round(_s,2))
Tp=Tp[:,_s>0.5]
G=F@Tp
print('dims',Tp.shape[1],'G sv',np.round(np.linalg.svd(G,compute_uv=False),2))
# Similarly induction composite in Tp
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
for (l,j) in [(2,1),(3,5),(3,1)]:
    B=Tp.T@QK(l,j)@OV0(5)@Tp
    print(f'L{l}H{j} induct composite sv in Tp',np.round(np.linalg.svd(B,compute_uv=False)/np.linalg.svd(B,compute_uv=False)[0],2))
print('OV0(5) on Tp sv',np.round(np.linalg.svd(OV0(5)@Tp,compute_uv=False),2))
```