```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]; mask=np.ones(384,bool); mask[sink]=False
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
Uo,So,Vto=np.linalg.svd(OV3)
T=Vto[:28].T
qb,kb,vb,ob=headmats(3,3)
OVb=ob@vb
Ub,Sb,Vtb=np.linalg.svd(OVb)
print('L3H3 OV sv',Sb[:30])
for kk in [5,10,15,20,24,28]:
    B=Vtb[:kk].T
    print(kk,'L3H3 V-read top dirs: frac in T28', np.round(np.linalg.norm(T.T@B,axis=0)**2,2))
# principal angles between T(18) and L3H3 read(18)
B=Vtb[:18].T; s=np.linalg.svd(Vto[:18]@B,compute_uv=False); print('cos principal angles T18 vs L3H3read18', np.round(s,2))
# L3H3 query read vs T
U,S,Vt=np.linalg.svd(qb.T@kb)
print('L3H3 QK top query dirs frac in T28', np.round(np.linalg.norm(T.T@U[:,:12],axis=0)**2,2))
print('L3H3 QK top key dirs frac in T28', np.round(np.linalg.norm(T.T@Vt[:12].T,axis=0)**2,2))
P=Uo[:,:28]
print('L3H3 QK top key dirs frac in P28 (prev-token out)', np.round(np.linalg.norm(P.T@Vt[:12].T,axis=0)**2,2))
```