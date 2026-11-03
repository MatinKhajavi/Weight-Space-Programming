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
# token directions in keys of sink heads
dirs={}
for h in [0,1,2,4,5]:
    q,k,v,o=headmats(0,h)
    U,S,Vt=np.linalg.svd(k)
    for i in range(3):
        d=T@(T.T@Vt[i]); 
        dirs[f'H{h}k{i}']=(d/np.linalg.norm(d), np.linalg.norm(T.T@Vt[i])**2)
names=list(dirs)
D=np.array([dirs[n][0] for n in names])
print('token-part fraction:', {n:round(dirs[n][1],2) for n in names})
print(names)
print(np.round(np.abs(D@D.T),2))
```