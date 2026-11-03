```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
w0=W['ln1_w.0']
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
def orth(A): 
    Q,_=np.linalg.qr(A); return Q
T3=rowspace(blk('W_v',0,3),20); Q3=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),30)
Tres=orth(np.diag(1/w0)@T3); Qres=orth(np.diag(1/w0)@Q3)
Tres2=orth(np.diag(w0)@T3); Qres2=orth(np.diag(w0)@Q3)
def frac(A,B): return np.linalg.norm(A@B)**2/np.linalg.norm(A)**2
print("w0 of top dims in T3:", np.sort(w0[np.argsort(-np.abs(T3).sum(1))[:20]]))
print("w0 of top dims in Q3:", np.sort(w0[np.argsort(-np.abs(Q3).sum(1))[:20]]))
print("random expectation: T 0.052, Q 0.078")
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        q=blk('W_q',l,j)@wl; k=blk('W_k',l,j)@wl; v=blk('W_v',l,j)@wl
        print(f"L{l}H{j}: q T={frac(q,Tres):.2f}/{frac(q,Tres2):.2f} Q={frac(q,Qres):.2f}/{frac(q,Qres2):.2f} | k T={frac(k,Tres):.2f}/{frac(k,Tres2):.2f} Q={frac(k,Qres):.2f}/{frac(k,Qres2):.2f} | v T={frac(v,Tres):.2f}/{frac(v,Tres2):.2f} Q={frac(v,Qres):.2f}/{frac(v,Qres2):.2f}")
    fc=W[f'W_fc.{l}']@np.diag(W[f'ln2_w.{l}'])
    print(f"MLP{l} fc: T={frac(fc,Tres):.2f}/{frac(fc,Tres2):.2f} Q={frac(fc,Qres):.2f}/{frac(fc,Qres2):.2f}")
# also: do OV outputs of L0 heads / MLP0 write into T or Q (residual space)?
for j in range(6):
    O=blk('W_o',0,j) if False else W['W_o.0'][:,j*hd:(j+1)*hd]
    print(f"L0H{j} W_o cols: T={frac(O.T,Tres):.2f}/{frac(O.T,Tres2):.2f} Q={frac(O.T,Qres):.2f}/{frac(O.T,Qres2):.2f}")
print("MLP0 proj cols:", frac(W['W_proj.0'].T,Tres), frac(W['W_proj.0'].T,Tres2), frac(W['W_proj.0'].T,Qres), frac(W['W_proj.0'].T,Qres2))
```