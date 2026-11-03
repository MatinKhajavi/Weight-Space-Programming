```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
Q,K,V,O=headmats(0,3); OV3=O@V
U,S,Vt=np.linalg.svd(Q.T@K)
Pos=np.linalg.qr(np.hstack([U[:,:20],Vt[:20].T])*1.0)[0][:,:24]   # ln-scaled coords
Q0,K0,V0,O0=headmats(0,0)
Uv,Sv,Vvt=np.linalg.svd(np.vstack([OV3/np.linalg.norm(OV3),(O0@V0)/np.linalg.norm(O0@V0)]))
print('combined OV input sv',Sv[:40].round(3))
Tok=Vvt[:28].T
# remove positional
Tok=Tok-Pos@(Pos.T@Tok); Tok=np.linalg.svd(Tok,full_matrices=False)[0][:,:28]
# neuron reading vectors in ln-scaled coords of layer0 input... use ln2_w.0
F=W['W_fc.0']*W['ln2_w.0']
# convert to ln1-scaled coords: raw residual r -> ln1 scaled is lw0*r; F reads raw*ln2w. Tok basis in ln1 coords: raw dir = Tok/lw0
Traw=np.linalg.qr(Tok/lw0[:,None])[0]
R=F@Traw
nr=np.linalg.norm(R,axis=1); print('neuron proj norms pct',np.percentile(nr,[10,50,90,99]).round(2), 'vs full', np.percentile(np.linalg.norm(F,axis=1),[10,50,90]).round(2))
np.save('/tmp/R.npy',R)
Rn=R/nr[:,None]
G=Rn@Rn.T; np.fill_diagonal(G,-1)
print('max cos to another neuron pct',np.percentile(G.max(1),[10,25,50,75,90]).round(2))
```