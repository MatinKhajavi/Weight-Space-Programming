```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]
mask=np.ones(384,bool); mask[sink]=False
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
OV0={h:(lambda q,k,v,o: o@v)(*headmats(0,h)) for h in range(6)}
for h in OV0: OV0[h][~mask]=0
# baseline by random orthogonal rotation of output space
Qr,_=np.linalg.qr(rng.standard_normal((384,384)))
for lb in [1,2,3]:
    print('layer',lb)
    for b in range(6):
        qb,kb,vb,ob=headmats(lb,b)
        qb=qb.copy(); kb=kb.copy(); vb=vb.copy()
        qb[:,~mask]=0; kb[:,~mask]=0; vb[:,~mask]=0
        s=[]
        for a in [0,3]:
            s.append(f'L0H{a}: K {comp(kb,OV0[a]):.3f} Q {comp(qb,OV0[a]):.3f} V {comp(vb,OV0[a]):.3f} | rnd K {comp(kb,Qr@OV0[a]):.3f}')
        mp=W['W_proj.0'].copy(); mp[~mask]=0
        s.append(f'MLP0: K {comp(kb,mp):.3f} Q {comp(qb,mp):.3f} V {comp(vb,mp):.3f}')
        print(f' H{b} ', ' || '.join(s))
```