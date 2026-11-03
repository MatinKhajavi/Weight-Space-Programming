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
Ssum=np.zeros((d,d)); mats={}
for (l2,h2) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(0,5)]; S=(M+M.T)/2; mats[f'ind{l2}{h2}']=S; Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
kT=24
T=U[:,::-1][:,:kT]
Ms=[]
names=[]
for k,S in mats.items():
    Ms.append(T.T@S@T); names.append(k)
for j in range(6):
    q,k,v,o=head(1,j); QK=q.T@k; Ms.append(T.T@((QK+QK.T)/2)@T); names.append(f'L1H{j}')
# add induction with L1 OVs too (context matching)
for (l2,h2),(l1,h1) in [((3,5),(1,3)),((2,1),(1,3)),((3,5),(2,2))]:
    q,k,v,o=head(l2,h2); M=(q.T@k)@OV[(l1,h1)]; S=(M+M.T)/2; Ms.append(T.T@S@T); names.append(f'ind{l2}{h2}<-{l1}{h1}')
Ms=[M/np.linalg.norm(M) for M in Ms]
# Jacobi-like joint diagonalization (Cardoso's JADE style, real symmetric)
def jd(Ms, iters=100):
    n=Ms[0].shape[0]; V=np.eye(n); Ms=[M.copy() for M in Ms]
    for it in range(iters):
        changed=0
        for p in range(n):
            for q in range(p+1,n):
                g=np.array([[M[p,p]-M[q,q], 2*M[p,q]] for M in Ms])
                G=g.T@g
                w,vecs=np.linalg.eigh(G); x,y=vecs[:,-1]
                if x<0: x,y=-x,-y
                c=np.sqrt((1+x)/2); s=y/(2*c)
                if abs(s)>1e-9:
                    changed+=1
                    R=np.eye(n); R[p,p]=c; R[q,q]=c; R[p,q]=-s; R[q,p]=s
                    Ms=[R.T@M@R for M in Ms]; V=V@R
        if changed==0: break
    return V,Ms
def offfrac(M): return 1-np.sum(np.diag(M)**2)/np.sum(M**2)
print('off-diagonal fraction in Ssum eigenbasis:',[f'{n}:{offfrac(M):.2f}' for n,M in zip(names,Ms)])
V,Md=jd(Ms)
print('after joint diag:',[f'{n}:{offfrac(M):.2f}' for n,M in zip(names,Md)])
# diagonals
D=np.array([np.diag(M) for M in Md])
print('diag values per basis vector (rows=matrices):')
print(np.round(D,2))
# A,B in this basis
q0,k0,v0,o0=head(0,0); u,s,vt=np.linalg.svd(k0@T,full_matrices=False); A=T@vt[0]
q3,k3,v3,o3=head(0,3); u,s,vt=np.linalg.svd(k3@T,full_matrices=False); B=T@vt[0]
B=B-A*(A@B); B/=np.linalg.norm(B)
basis=T@V
print('A coords:',np.round(basis.T@A,2)); print('B coords:',np.round(basis.T@B,2))
# random baseline for joint diag quality: random symmetric matrices
rng=np.random.default_rng(0)
Rm=[]
for _ in range(len(Ms)):
    X=rng.standard_normal((kT,kT)); X=(X+X.T)/2; Rm.append(X/np.linalg.norm(X))
Vr,Rd=jd(Rm); print('random matrices after joint diag offfrac:',np.round([offfrac(M) for M in Rd],2))
```