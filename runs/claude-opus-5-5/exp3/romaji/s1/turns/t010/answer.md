```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
def planes(l,h,n=8, raw=False):
    s=slice(64*h,64*h+64)
    if raw:
        q=W[f'W_q.{l}'][s]; k=W[f'W_k.{l}'][s]
    else:
        q,k,_,_=headmats(l,h)
    ev,Y=np.linalg.eig(k@q.T)
    idx=np.argsort(-np.abs(ev))
    out=[]
    used=set()
    for i in idx:
        if len(out)>=n: break
        e=ev[i]
        if e.imag<-1e-9: continue
        x=q.T@Y[:,i]
        if abs(e.imag)>1e-9:
            B=np.stack([x.real,x.imag],1)
        else:
            B=x.real[:,None]
        Qb,_=np.linalg.qr(B)
        out.append((e,Qb))
    return out
P3=planes(0,3,8)
for l,h in [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(0,0)]:
    Pl=planes(l,h,6)
    print(f'L{l}H{h}')
    for e,Qb in Pl:
        ov=[np.linalg.norm(Qb.T@Q3)/np.sqrt(min(Qb.shape[1],Q3.shape[1])) for _,Q3 in P3]
        print(f'   {abs(e):.2f}@{np.degrees(np.angle(e)):.0f}  overlap with L0H3 planes:',np.round(ov,2))
print('L0H3 planes', [f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e,_ in P3])
```