```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]; mask=np.ones(384,bool); mask[sink]=False
M=W['W_proj.3'][mask]
U,S,Vt=np.linalg.svd(M)
ks=[15,20,21,22,23,24,25,26,27,28,30,35,40,60]
print('k:',ks, ' random expectation k/379')
for h in range(6):
    q,k,v,o=headmats(3,h); OV=(o@v)[mask]
    fr=[np.linalg.norm(U[:,:kk].T@OV)**2/np.linalg.norm(OV)**2 for kk in ks]
    print(f'L3H{h}',np.round(fr,3))
for l in [2,1]:
  for h in range(6):
    q,k,v,o=headmats(l,h); OV=(o@v)[mask]
    fr=[np.linalg.norm(U[:,:kk].T@OV)**2/np.linalg.norm(OV)**2 for kk in ks]
    print(f'L{l}H{h}',np.round(fr,3))
M2=W['W_proj.2'][mask]
print('W_proj.2', np.round([np.linalg.norm(U[:,:kk].T@M2)**2/np.linalg.norm(M2)**2 for kk in ks],3))
```