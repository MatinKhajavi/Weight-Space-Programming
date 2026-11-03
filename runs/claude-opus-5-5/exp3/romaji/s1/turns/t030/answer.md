```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]; mask=np.ones(384,bool); mask[sink]=False
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
Uo,So,Vto=np.linalg.svd(OV3)
T=Vto[:28].T
q5,k5,v5,o5=headmats(0,5)
d=T@(T.T@np.linalg.svd(k5)[2][0]); d/=np.linalg.norm(d)
U3,S3,_=np.linalg.svd(W['W_proj.3'][mask])
qb,kb,vb,ob=headmats(3,3); C=(ob@vb)[mask]
def prof(x):
    y=U3[:,:30].T@(C@x)
    return y
y=prof(d)
print('d* through L3H3 copy, coefficients on W_proj.3 sing dirs 0..29:')
print(np.round(np.abs(y)/np.linalg.norm(C@d),2))
# baseline: average profile over T basis vectors
Y=np.array([np.abs(prof(T[:,i]))/np.linalg.norm(C@T[:,i]) for i in range(28)])
print('avg over T basis:'); print(np.round(Y.mean(0),2))
print('fraction of C@d in top-27 readout', np.linalg.norm(U3[:,:27].T@(C@d))**2/np.linalg.norm(C@d)**2, ' avg T basis', np.mean([np.linalg.norm(U3[:,:27].T@(C@T[:,i]))**2/np.linalg.norm(C@T[:,i])**2 for i in range(28)]))
# also how does d* map through L0H3 (prev-token) -- norm of OV3 d relative
print('|OV3 d*|',np.linalg.norm(OV3@d),' singular values of OV3 top', So[:5])
print('d* coords on OV3 right singular vectors', np.round(Vto[:28]@d,2))
```