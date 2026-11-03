```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
lw0=W['ln1_w.0']
L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
outs=[]
for h in range(6):
    sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']
    u,s,vt=np.linalg.svd(OV,full_matrices=False); outs.append(u[:,:{0:3,4:3,1:27,5:27,2:10,3:10}[h]])
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); alive=np.where(an>0.8)[0]; Aa=A[alive]
S=np.hstack([P]+outs); Qs,_=np.linalg.qr(S)
R=Aa-(Aa@Qs)@Qs.T
u,s,vt=np.linalg.svd(R,full_matrices=False)
print("sv of W_fc.0 alive rows after removing P and head outputs:", s[:50])
Tp=vt[:28].T
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
print("overlap T' (28) with T (30):", np.linalg.norm(Tp.T@T)**2, "of max 28")
# fraction energy of various matrices in T'
def fr(M,S): return np.linalg.norm(M@S)**2/np.linalg.norm(M)**2
print("W_fc.0 alive in T':", fr(Aa,Tp), " in T:", fr(Aa,T))
for h in range(6):
    print(f"L0H{h} v in T': {fr(L0[f'{h}v'],Tp):.2f}  q: {fr(L0[f'{h}q'],Tp):.2f}  k: {fr(L0[f'{h}k'],Tp):.2f}")
lw=W['ln1_w.1']
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    print(f"L1H{j} q in T': {fr(W['W_q.1'][sl]*lw,Tp):.2f} k: {fr(W['W_k.1'][sl]*lw,Tp):.2f} v: {fr(W['W_v.1'][sl]*lw,Tp):.2f}")
```