```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
lw0=W['ln1_w.0']; L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
outs=[]
for h in range(6):
    sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']
    u,s,vt=np.linalg.svd(OV,full_matrices=False); outs.append(u[:,:{0:3,4:3,1:27,5:27,2:10,3:10}[h]])
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
lw=W['ln1_w.1']
Ks=np.vstack([W['W_k.1'][j*hd:(j+1)*hd]*lw[None,:] for j in range(6)])
Qs,_=np.linalg.qr(np.hstack([P,outs[0],outs[4]]))
Kr=Ks-(Ks@Qs)@Qs.T
u,s,vt=np.linalg.svd(Kr,full_matrices=False)
print("L1 keys residual sv:", s[:12])
tstar=vt[0]; 
print("t* energy in T:", np.linalg.norm(tstar@T)**2, " in P:", np.linalg.norm(tstar@P)**2, " in H1o:", np.linalg.norm(tstar@outs[1])**2, "H5o:", np.linalg.norm(tstar@outs[5])**2, "H2o", np.linalg.norm(tstar@outs[2])**2, "H3o", np.linalg.norm(tstar@outs[3])**2)
for i in range(1,4):
    v=vt[i]; print(f"dir{i} energy in T:", np.linalg.norm(v@T)**2, " P:", np.linalg.norm(v@P)**2, " H1o:", np.linalg.norm(v@outs[1])**2, "H5o:", np.linalg.norm(v@outs[5])**2)
# per-head: energy of each head's K along t*
for j in range(6):
    K=W['W_k.1'][j*hd:(j+1)*hd]*lw[None,:]
    print(f"L1H{j} K energy along t*: {np.linalg.norm(K@tstar)**2/np.linalg.norm(K)**2:.3f}; along dir1: {np.linalg.norm(K@vt[1])**2/np.linalg.norm(K)**2:.3f}")
# who else reads t*? layer-0 q/k, MLP0 rows, MLP1 rows, layer-2/3 q/k
for h in range(6):
    print(f"L0H{h}: q {np.linalg.norm(L0[f'{h}q']@tstar)**2/np.linalg.norm(L0[f'{h}q'])**2:.3f} k {np.linalg.norm(L0[f'{h}k']@tstar)**2/np.linalg.norm(L0[f'{h}k'])**2:.3f} v {np.linalg.norm(L0[f'{h}v']@tstar)**2/np.linalg.norm(L0[f'{h}v'])**2:.3f}")
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); alive=an>0.8
c=(A[alive]@tstar)/an[alive]
print("MLP0 alive neurons cos with t*: hist", np.histogram(c,bins=np.linspace(-1,1,11))[0], " >0.4:", (c>0.4).sum(), " <-0.4:", (c<-0.4).sum())
A1=W['W_fc.1']*W['ln2_w.1'][None,:]; c1=(A1@tstar)/np.linalg.norm(A1,axis=1)
print("MLP1 neurons cos with t*: hist", np.histogram(c1,bins=np.linspace(-1,1,11))[0])
# is t* roughly in the token embedding? check alignment with the top-cluster direction of MLP0
print("baseline 1/384:", 1/384)
```