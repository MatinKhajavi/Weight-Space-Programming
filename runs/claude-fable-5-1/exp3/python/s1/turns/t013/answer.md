```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
Ssum=np.zeros((d,d)); Ms={}
for (l2,h2) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(0,5)]; S=(M+M.T)/2; Ms[(l2,h2)]=S; Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
T=U[:,::-1][:,:22]; PT=T@T.T
q0,k0,v0,o0=head(0,0); u,s,vt=np.linalg.svd(k0@T,full_matrices=False); A=T@vt[0]
q3,k3,v3,o3=head(0,3); u,s,vt=np.linalg.svd(k3@T,full_matrices=False); B=T@vt[0]
B=B-A*(A@B); B/=np.linalg.norm(B)
print('Ssum eigen top:',ev[::-1][:22])
print('A^T Ssum A =',A@Ssum@A,' B^T Ssum B =',B@Ssum@B)
# is A an eigenvector? residual of Ssum A
for nm,x in [('A',A),('B',B)]:
    y=Ssum@x; print(nm,'Rayleigh',x@y,'|Ssum x|',np.linalg.norm(y),'cos(x,Ssum x)',x@y/np.linalg.norm(y))
    for key,S in Ms.items():
        print('   ',key,'x^T S x=',x@S@x,' |S| top eig', np.linalg.eigvalsh(S)[-1], 'median diag on T', np.median(np.diag(T.T@S@T)))
# typical values for random unit directions in T
rng=np.random.default_rng(0)
vals=[]
for _ in range(200):
    r=T@rng.standard_normal(22); r/=np.linalg.norm(r); vals.append(r@Ssum@r)
print('random dir in T: Rayleigh mean',np.mean(vals),'std',np.std(vals))
# OV of prev-token head applied to A, B vs random T dirs
for (l,j) in [(0,5),(0,1),(2,1),(3,1),(3,5),(0,0),(0,4)]:
    M=OV[(l,j)]
    rv=[np.linalg.norm(M@(T@rng.standard_normal(22))/np.sqrt(22)) for _ in range(100)]
    print(f'OV L{l}H{j}: |OV A|={np.linalg.norm(M@A):.2f} |OV B|={np.linalg.norm(M@B):.2f} random T dir mean {np.mean(rv):.2f}')
# MLP0 and MLP1 read A,B?
for l in range(4):
    F=W[f'W_fc.{l}']
    rv=[np.linalg.norm(F@(T@rng.standard_normal(22))/np.sqrt(22)) for _ in range(100)]
    print(f'W_fc.{l}: |F A|={np.linalg.norm(F@A):.2f} |F B|={np.linalg.norm(F@B):.2f} random T dir {np.mean(rv):.2f}; #neurons |w.A|>0.3: {(abs(F@A)>0.3).sum()} |w.B|>0.3: {(abs(F@B)>0.3).sum()}')
```