```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
g=W['ln1_w.0']
def headmats(l,j):
    gg=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*gg, W[f'W_k.{l}'][s]*gg, W[f'W_v.{l}'][s]*gg, W[f'W_o.{l}'][:,s]
feat={}
for j in range(6):
    Wq,Wk,Wv,Wo=headmats(0,j)
    A=Wq.T@Wk; OV=Wo@Wv
    feat[f'H{j}k']=(A**2).sum(0); feat[f'H{j}q']=(A**2).sum(1); feat[f'H{j}v']=(OV**2).sum(0)
feat['g']=g**2
feat['fc0']=((W['W_fc.0']*W['ln2_w.0'])**2).sum(0)
names=list(feat.keys())
F=np.array([feat[n]/feat[n].sum() for n in names])
C=np.corrcoef(np.log(F+1e-9))
print(names)
print(C.round(2))
# top coordinates of H2k vs H2v
print('H2k top coords',np.argsort(-feat['H2k'])[:30])
print('H2v top coords',np.argsort(-feat['H2v'])[:30])
print('broad v top', np.argsort(-feat['H0v'])[:30])
# participation ratio
for n in names: f=F[names.index(n)]; print(n, 'PR %.1f'%(1/(f**2).sum()))
```