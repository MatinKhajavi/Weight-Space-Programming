```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]; w2=W["ln2_w.0"]; w1=W["ln1_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
E = rsv(Fcomp,28)
FE = F@E; nrm=np.linalg.norm(FE,axis=1); frac = nrm**2/(np.linalg.norm(F,axis=1)**2)
sel = np.where(frac>0.5)[0]
X = FE[sel]/nrm[sel][:,None]
D = 1-np.abs(X@X.T); D=(D+D.T)/2; np.fill_diagonal(D,0)
Z = linkage(squareform(D,checks=False), method='average')
lab = fcluster(Z, t=0.4, criterion='distance')
cents=[]; members=[]
for c in range(1,lab.max()+1):
    idx=np.where(lab==c)[0]
    if len(idx)<2: continue
    V=X[idx]; sg=np.sign(V@V[0]); V=V*sg[:,None]; m=V.mean(0); m/=np.linalg.norm(m); cents.append(m); members.append((sel[idx],sg))
C = E@np.array(cents).T
A = F@C
sg = np.sign((np.maximum(A,0)**2).sum(0)-(np.maximum(-A,0)**2).sum(0)); C*=sg
members=[(idx,s*g) for (idx,s),g in zip(members,sg)]
k=C.shape[1]
z = C/np.clip(w2,0.2,None)[:,None]
hz = z*w1[:,None]   # layer-0 LN1 output direction (unnormalized)
print("token idx:           ", np.arange(k))
print("|h1| (ln1-scaled norm):", np.linalg.norm(hz,axis=0).round(2))
for j in range(6):
    s=slice(64*j,64*(j+1))
    vv = np.linalg.norm(W["W_v.0"][s]@hz,axis=0)
    qq = np.linalg.norm(W["W_q.0"][s]@hz,axis=0); kk=np.linalg.norm(W["W_k.0"][s]@hz,axis=0)
    print(f"L0H{j} |V h_t|:", vv.round(2), " |K h_t| mean %.2f max %.2f(tok %d)" % (kk.mean(), kk.max(), kk.argmax()))
# MLP0 per-token-cluster output norm and number of neurons; also the mean |W_proj column| of member neurons
Pj=W["W_proj.0"]
print("cluster #neurons:", [len(m[0]) for m in members])
print("cluster mean |proj col|:", np.array([np.linalg.norm(Pj[:,m[0]],axis=0).mean() for m in members]).round(2))
print("cluster mean |fc row| :", np.array([np.linalg.norm(F[m[0]],axis=1).mean() for m in members]).round(2))
# layer-1 Q/K/V reading of the token direction (through ln1_w.1)
h1l = z*W["ln1_w.1"][:,None]
for nm in ["W_q.1","W_k.1","W_v.1"]:
    r = np.linalg.norm(W[nm]@h1l,axis=0)
    print(nm, "|.h_t| :", r.round(1))
# L0H2 output for token t (prev-token signal) magnitude and L1 Q reading of it
u2 = OVe[(0,2)]@z     # 384 x k
print("|OV2 z_t|:", np.linalg.norm(u2,axis=0).round(2))
print("W_q.1 |. (OV2 z_t * ln1_w.1)|:", np.linalg.norm(W["W_q.1"]@(u2*W["ln1_w.1"][:,None]),axis=0).round(1))
```