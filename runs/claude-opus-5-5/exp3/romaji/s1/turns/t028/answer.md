```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]; mask=np.ones(384,bool); mask[sink]=False
f=lambda e: e.real.sum()/np.abs(e).sum()
for src in [1,2]:
    OVs=[]
    for a in range(6):
        q,k,v,o=headmats(src,a); M=o@v; M[sink]=0; OVs.append(M)
    for lb in range(src+1,4):
        for b in range(6):
            qb,kb,vb,ob=headmats(lb,b)
            qb=qb.copy(); kb=kb.copy(); qb[:,sink]=0; kb[:,sink]=0
            out=[]
            for a in range(6):
                # query reads OV_a output at query position, key reads OV_a output at key position:  OV_a^T A OV_a
                B=OVs[a].T@qb.T@kb@OVs[a]
                ev=np.linalg.eigvals(kb@OVs[a]@OVs[a].T@qb.T)  # same nonzero eigs as B
                out.append(f'{f(ev):+.2f}')
            # also total L1-out subspace matching
            print(f'L{src}->L{lb}H{b}: same-feature match positivity per src head', ' '.join(out))
```