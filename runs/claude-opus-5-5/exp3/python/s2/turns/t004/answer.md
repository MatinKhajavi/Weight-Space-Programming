```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T, s
Vall,s=rs(W['W_v.0']*w0,30)
heads={}
for j in range(6):
    sl=slice(64*j,64*j+64)
    heads[j]=dict(Q=W['W_q.0'][sl]*w0,K=W['W_k.0'][sl]*w0,V=W['W_v.0'][sl]*w0,O=W['W_o.0'][:,sl])
def angles(A,B):
    return np.linalg.svd(A.T@B,compute_uv=False)
for j in range(6):
    Vj,sv=rs(heads[j]['V'],28)
    print(j,'V sv',np.round(sv[:34],2))
    print('   cos with Vall30', np.round(angles(Vj[:,:20],Vall),2)[[0,5,10,15,19]])
# token subspace candidate: V of heads 1 and 5
V1,_=rs(heads[1]['V'],28);V5,_=rs(heads[5]['V'],28)
print('V1 vs V5', np.round(angles(V1,V5),2))
K1,sk=rs(heads[1]['K'],40);K5,sk5=rs(heads[5]['K'],40)
Q1,sq=rs(heads[1]['Q'],40);Q5,sq5=rs(heads[5]['Q'],40)
print('K1 sv',np.round(sk,2));print('Q1 sv',np.round(sq,2));print('K5 sv',np.round(sk5,2));print('Q5 sv',np.round(sq5,2))
print('K1 vs V1',np.round(angles(K1[:,:20],V1),2))
print('Q1 vs V1',np.round(angles(Q1[:,:20],V1),2))
print('K5 vs V5',np.round(angles(K5[:,:20],V5),2))
print('Q5 vs V5',np.round(angles(Q5[:,:20],V5),2))
```