```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0
_,_,Vt=np.linalg.svd(V3); T=Vt[:24].T
OV3=W['W_o.0'][:,192:256]@V3; Uo,So,_=np.linalg.svd(OV3); H3out=Uo[:,:20]
OV5=W['W_o.0'][:,320:384]@(hm(0,'v',5)*ln0); Uo,So,_=np.linalg.svd(OV5); H5rich=Uo[:,2:8]; H5pos=Uo[:,:2]
Fc=W['W_fc.0']*W['ln2_w.0']
tot=np.linalg.norm(Fc,axis=1)**2
fT=np.linalg.norm(Fc@T,axis=1)**2/tot
fP=np.linalg.norm(Fc@H3out,axis=1)**2/tot
fR=np.linalg.norm(Fc@H5rich,axis=1)**2/tot
fPos=np.linalg.norm(Fc@H5pos,axis=1)**2/tot
print("MLP0 neuron input fractions: T(cur) mean %.3f, H3out(prev) mean %.3f, H5rich mean %.3f, H5pos mean %.3f"%(fT.mean(),fP.mean(),fR.mean(),fPos.mean()))
print("chance: T 24/384=%.3f, H3out 20/384=%.3f, H5rich 6/384=%.3f, H5pos 2/384=%.3f"%(24/384,20/384,6/384,2/384))
# joint distribution
import pandas as pd
df=pd.DataFrame(dict(T=fT,prev=fP,rich=fR,norm=np.sqrt(tot)))
print(df.describe())
print("corr T-prev:",np.corrcoef(fT,fP)[0,1])
# neurons with high prev and high T
print("frac neurons with prev>0.2:",(fP>0.2).mean(),"T>0.2:",(fT>0.2).mean(),"both>0.15:",((fP>0.15)&(fT>0.15)).mean())
# Output side: where do MLP0 neurons write? Fraction into T-like subspace (token id) vs others; compare with W_proj.0 read by later Q/K
Pj=W['W_proj.0']
print("W_proj.0 column norms: mean %.3f, max %.3f, frac of dead-ish (norm<0.2): %.3f"%(np.linalg.norm(Pj,axis=0).mean(),np.linalg.norm(Pj,axis=0).max(),(np.linalg.norm(Pj,axis=0)<0.2).mean()))
print("W_fc.0 row norms: mean %.3f, min %.3f, max %.3f"%(np.linalg.norm(W['W_fc.0'],axis=1).mean(),np.linalg.norm(W['W_fc.0'],axis=1).min(),np.linalg.norm(W['W_fc.0'],axis=1).max()))
```