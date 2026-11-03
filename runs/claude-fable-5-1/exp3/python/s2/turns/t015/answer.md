```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
lw0=W['ln1_w.0']
L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
outs=[]
for h in range(6):
    sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']
    u,s,vt=np.linalg.svd(OV,full_matrices=False); outs.append(u[:,:(27 if h in (1,2,3,5) else 3)])
# orthogonalize sequentially: T, H1out, H5out, H0out, H4out, P, H2out, H3out
order=[('T',T),('H1o',outs[1]),('H5o',outs[5]),('H0o',outs[0]),('H4o',outs[4]),('P',P),('H2o',outs[2]),('H3o',outs[3])]
basis=[];labels=[]
Qacc=np.zeros((384,0))
for name,S in order:
    R = S - Qacc@(Qacc.T@S)
    q,_=np.linalg.qr(R)
    Qacc=np.hstack([Qacc,q]); basis.append(q); labels.append(name)
rest,_=np.linalg.qr(np.eye(384)-Qacc@Qacc.T); 
# take complement basis
u,s,vt=np.linalg.svd(np.eye(384)-Qacc@Qacc.T); rest=u[:,:384-Qacc.shape[1]]
basis.append(rest); labels.append('rest')
print("dims:", [b.shape[1] for b in basis])
import pandas as pd
pd.set_option('display.width',250)
for l in range(1,4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        Q=W[f'W_q.{l}'][sl]*lw[None,:]; K=W[f'W_k.{l}'][sl]*lw[None,:]
        QK=Q.T@K
        tot=np.linalg.norm(QK)**2
        M=np.zeros((len(basis),len(basis)))
        for a,Ba in enumerate(basis):
            for b,Bb in enumerate(basis):
                M[a,b]=np.linalg.norm(Ba.T@QK@Bb)**2/tot
        df=pd.DataFrame(M,index=['q:'+x for x in labels],columns=['k:'+x for x in labels])
        print(f"L{l}H{j} QK block energy fractions (trace/fro {np.trace(QK)/np.linalg.norm(QK):.2f}):")
        print(df.round(3).to_string())
```