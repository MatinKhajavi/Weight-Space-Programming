```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
rng=np.random.default_rng(0)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K/8,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
def sv(A): return np.linalg.svd(A)
lw0=W['ln1_w.0']
U,S,Vt=sv(H[0][3]['OV']); Tok=np.linalg.qr(Vt[:20].T/lw0[:,None])[0]
def ovraw(h): 
    s=slice(64*h,64*h+64); return W['W_o.0'][:,s]@W['W_v.0'][s]*lw0
QK=H[3][3]['QK']
for name,src in [('L0H3',ovraw(3)),('L0H0',ovraw(0)),('L0H5',ovraw(5)),('identity',np.eye(384))]:
    M=Tok.T@QK@src@Tok; ev=np.linalg.eigvals(M)
    print(name,'tr/fro %.2f'%(np.trace(M)/np.linalg.norm(M)),'fro %.3f'%np.linalg.norm(M),'realfrac %.2f'%(ev.real.sum()/np.abs(ev).sum()))
# random token subspaces baseline
vals=[]
for t in range(200):
    R=np.linalg.qr(rng.standard_normal((384,20)))[0]
    M=R.T@QK@ovraw(3)@R; vals.append(np.trace(M)/np.linalg.norm(M))
print('random subspace baseline tr/fro mean %.2f sd %.2f max %.2f'%(np.mean(vals),np.std(vals),np.max(vals)))
# same test for every head with L0H3 composed
for l in [1,2,3]:
    print(l,[round(float(np.trace(Tok.T@H[l][h]['QK']@ovraw(3)@Tok)/np.linalg.norm(Tok.T@H[l][h]['QK']@ovraw(3)@Tok)),2) for h in range(6)])
# does L3H3 OV read token identity?
for l in [1,2,3]:
    print('OV reads Tok frac', l, [round(float(np.linalg.norm(H[l][h]['OV']@Tok)/np.linalg.norm(H[l][h]['OV'])),2) for h in range(6)])
```