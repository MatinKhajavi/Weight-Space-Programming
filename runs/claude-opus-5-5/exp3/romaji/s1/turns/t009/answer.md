```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
# output directions of L0 heads
outs={}
for h in range(6):
    q,k,v,o=headmats(0,h)
    u,s,vt=np.linalg.svd(o@v)
    outs[h]=(u,s)
for l in range(1,4):
    b=W[f'ln1_b.{l}']; w=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]
        qb=Wq@b
        # typical query size: frobenius of Wq*w times 1 (unit-variance normalized input -> norm ~ ||Wq*w||_F)
        qtyp=np.linalg.norm(Wq*w)
        r=(Wk.T@qb)*w   # key direction (in normalized residual coords) scored by constant query
        rn=r/np.linalg.norm(r)
        # alignment with L0 head top output dirs
        al=[float(np.linalg.norm(outs[a][0][:,:2].T@rn)) for a in range(6)]
        print(f'L{l}H{h} |qbias|={np.linalg.norm(qb):.2f} typ|q|~{qtyp:.2f} |r|={np.linalg.norm(r):.2f}  align r with L0 top2 out dirs:',np.round(al,2))
```