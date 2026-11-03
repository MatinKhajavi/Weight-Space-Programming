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
u,s,vt=np.linalg.svd(Kr,full_matrices=False); tstar=vt[0]
B=W['W_proj.0']; bn=np.linalg.norm(B,axis=0)
proj=B.T@tstar   # per-neuron output along t*
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1)
print("t* energy in W_proj.0 column space (top-k left sv):")
uB,sB,vB=np.linalg.svd(B,full_matrices=False)
cum=np.cumsum((uB.T@tstar)**2); print(cum[[4,9,19,49,99,199]])
idx=np.argsort(-np.abs(proj))[:25]
print("top neurons writing along t*: idx, proj, out-norm, in-norm, cos(out,t*)")
for i in idx: print(i, round(float(proj[i]),3), round(float(bn[i]),2), round(float(an[i]),2), round(float(proj[i]/bn[i]),2))
# what do these top neurons read? cluster membership via cosine among their input vectors (in T coords & raw)
top=idx[:12]
Ain=A[top]; Ain_n=Ain/np.linalg.norm(Ain,axis=1)[:,None]
print("input cosines among top-12 t* neurons (raw 384):")
print(np.round(Ain_n@Ain_n.T,2))
# their reading decomposition
def fr(M,S): return np.linalg.norm(M@S)**2/np.linalg.norm(M)**2
for i in top:
    a=A[i:i+1]
    print(i, f"T {fr(a,T):.2f} P {fr(a,P):.2f} H1o {fr(a,outs[1]):.2f} H5o {fr(a,outs[5]):.2f} H0o {fr(a,outs[0]):.2f} H4o {fr(a,outs[4]):.2f} H2o {fr(a,outs[2]):.2f} H3o {fr(a,outs[3]):.2f}")
```