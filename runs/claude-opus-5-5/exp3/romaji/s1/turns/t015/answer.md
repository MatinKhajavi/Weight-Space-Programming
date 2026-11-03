```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]
mask=np.ones(384,bool); mask[sink]=False
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
q0,k0,v0,o0=headmats(0,0); OV00=o0@v0
rng=np.random.default_rng(1)
for src,OV in [('L0H3',OV3),('L0H0',OV00)]:
  print('source',src)
  for lb in [1,2,3]:
    for b in range(6):
        qb,kb,vb,ob=headmats(lb,b)
        kb=kb.copy(); kb[:,~mask]=0; qb=qb.copy(); qb[:,~mask]=0
        # 64x64 matrix with same nonzero eigs:  kb OV qb^T
        ev=np.linalg.eigvals(kb@OV@qb.T)
        ev2=np.linalg.eigvals(qb@OV@kb.T)  # query reads moved token, key reads raw (reverse)
        f=lambda e: e.real.sum()/np.abs(e).sum()
        top=ev[np.argsort(-np.abs(ev))[:3]]
        print(f'  L{lb}H{b}: K<-src  pos-frac {f(ev):+.2f}  top {np.round(top,2)} |  Q<-src pos-frac {f(ev2):+.2f}')
```