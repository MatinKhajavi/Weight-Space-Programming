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
S=np.hstack([P]+outs); Qs,_=np.linalg.qr(S)
def clean(M): 
    R=M-(M@Qs)@Qs.T; return R/np.linalg.norm(R)
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); Aa=A[an>0.8]
blocks=[clean(Aa)*3]
for h in [1,5,2,3]: blocks.append(clean(L0[f'{h}v']))
lw=W['ln1_w.1']
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    blocks.append(clean(W['W_q.1'][sl]*lw)); blocks.append(clean(W['W_v.1'][sl]*lw))
A1=W['W_fc.1']*W['ln2_w.1'][None,:]; blocks.append(clean(A1)*2)
X=np.vstack(blocks)
u,s,vt=np.linalg.svd(X,full_matrices=False)
print("stacked sv:", s[:60])
Tok=vt[:28].T
np.save('/tmp/Tok.npy',Tok)
def fr(M,S): return np.linalg.norm(M@S)**2/np.linalg.norm(M)**2
print("W_fc.0 alive in Tok:", fr(Aa,Tok))
for h in range(6):
    print(f"L0H{h} v in Tok: {fr(L0[f'{h}v'],Tok):.2f}  q: {fr(L0[f'{h}q'],Tok):.2f}  k: {fr(L0[f'{h}k'],Tok):.2f}")
for l in range(1,4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        print(f"L{l}H{j} q in Tok: {fr(W[f'W_q.{l}'][sl]*lw,Tok):.2f} k: {fr(W[f'W_k.{l}'][sl]*lw,Tok):.2f} v: {fr(W[f'W_v.{l}'][sl]*lw,Tok):.2f}")
for l in range(4):
    print(f"W_fc.{l} in Tok:", fr(W[f'W_fc.{l}']*W[f'ln2_w.{l}'][None,:],Tok))
```