```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W["ln1_w.0"]
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
def lsv(A,r):
    u,_,_=np.linalg.svd(A,full_matrices=False); return u[:,:r]
def energy(A,S): return np.linalg.norm(A@S)**2/np.linalg.norm(A)**2
Q2=eff("W_q.0",2); K2=eff("W_k.0",2); V2=eff("W_v.0",2)
P = rsv(np.vstack([Q2,K2]),24); T = rsv(V2,30)
OVe = {j: W["W_o.0"][:,64*j:64*(j+1)]@eff("W_v.0",j) for j in range(6)}
Ub,_ = np.linalg.qr(np.hstack([lsv(OVe[j],3) for j in [0,1,4,5]]))
U2 = lsv(OVe[2],24)
F = W["W_fc.0"]*W["ln2_w.0"]
Sall,_ = np.linalg.qr(np.hstack([P,T,Ub,U2]))
Fres = F - (F@Sall)@Sall.T
R = rsv(Fres,30)
# full layer0 attention output subspace
OVall = np.hstack([OVe[j] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False)
print("layer0 total OV output sv:", s[:50])
for k in [30,60,100]:
    Uo = u[:,:k]
    print(f"R energy in top-{k} L0-attn-output subspace: {energy(R.T,Uo):.2f} (baseline {k/384:.2f});  T: {energy(T.T,Uo):.2f} P: {energy(P.T,Uo):.2f}")
# complement of attention output: the embedding-only directions. Which subspace does MLP0 read that attention doesn't write?
Uo = u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
print("W_fc.0 energy outside top-100 attn-output subspace: %.2f" % (np.linalg.norm(Fcomp)**2/np.linalg.norm(F)**2))
E = rsv(Fcomp,40)   # 'embedding-only' subspace read by MLP0
print("E sv:", np.linalg.svd(Fcomp,compute_uv=False)[:40])
print("E on ln1_w.0 small dims (<0.15): %.2f" % ((E[w0<0.15]**2).sum()/(E**2).sum()))
print("E vs R overlap:", np.linalg.svd(E.T@R,compute_uv=False)[:8])
print("E vs P overlap:", np.linalg.svd(E.T@P,compute_uv=False)[:8])
print("E vs T overlap:", np.linalg.svd(E.T@T,compute_uv=False)[:8])
# does any attention read E?
for l in range(4):
    row=[]
    for j in range(6):
        q=eff(f"W_q.{l}",j,l); k=eff(f"W_k.{l}",j,l); v=eff(f"W_v.{l}",j,l)
        row.append((energy(q,E),energy(k,E),energy(v,E)))
    print(f"layer {l} (Q,K,V) energy in E (baseline 0.10):", np.array(row).round(2).tolist())
# raw (unscaled) reads of E by layer-0 attention
for j in range(6):
    print(f"L0H{j} RAW energy in E: Q {energy(W['W_q.0'][64*j:64*j+64],E):.2f} K {energy(W['W_k.0'][64*j:64*j+64],E):.2f} V {energy(W['W_v.0'][64*j:64*j+64],E):.2f}")
```