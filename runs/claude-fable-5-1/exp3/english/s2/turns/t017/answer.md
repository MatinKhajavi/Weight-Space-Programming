```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
Uf,Sf,Vf=np.linalg.svd((fc*g2)[strong],full_matrices=False); T=Vf[:30].T
s2=slice(128,192)
U,S,Vt=np.linalg.svd((W["W_q.0"][s2]*g).T@(W["W_k.0"][s2]*g))
Uc,Sc,_=np.linalg.svd(np.concatenate([U[:,:50],Vt[:50].T],axis=1)); P=Uc[:,:60]
def OV(l,j):
    s=slice(j*hd,(j+1)*hd); gl=W[f"ln1_w.{l}"]; return W[f"W_o.{l}"][:,s]@(W[f"W_v.{l}"][s]*gl)
def topcols(M,k): u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
Ob=topcols(np.concatenate([OV(0,j) for j in [0,1,4,5]],axis=1),30)
O2=topcols(OV(0,2),40); O3=topcols(OV(0,3),30)
M0=topcols(W["W_proj.0"]@(W["W_fc.0"]*g2)[strong].T if False else W["W_proj.0"][:,strong],60)
# Gram-Schmidt sequence of subspaces
blocks=[("P",P),("T",T),("Ob",Ob),("O2",O2),("O3",O3),("M0",M0)]
basis=[]; names=[]
Q=np.zeros((d,0))
for nm,B in blocks:
    R=B-Q@(Q.T@B)
    u,s,vt=np.linalg.svd(R,full_matrices=False)
    keep=u[:,s>0.3]   # keep directions not already covered
    Q=np.concatenate([Q,keep],axis=1); names+= [nm]*keep.shape[1]
    print(nm, "new dims", keep.shape[1])
# rest
u,s,vt=np.linalg.svd(np.eye(d)-Q@Q.T); rest=u[:,s>0.5]; Q=np.concatenate([Q,rest],axis=1); names+=["rest"]*rest.shape[1]
names=np.array(names); print("total", Q.shape[1])
labs=["P","T","Ob","O2","O3","M0","rest"]
def blockmap(QK):
    A=Q.T@QK@Q; tot=np.sum(A**2)
    return np.array([[np.sum(A[np.ix_(names==a,names==b)]**2)/tot for b in labs] for a in labs])
for l in [1,2,3]:
    gl=W[f"ln1_w.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd); QK=(W[f"W_q.{l}"][s]*gl).T@(W[f"W_k.{l}"][s]*gl)
        B=blockmap(QK)
        if l==1 or j in [0,1]:
            print(f"L{l}H{j} QK energy blocks (rows query, cols key) labs {labs}")
            print(B)
```