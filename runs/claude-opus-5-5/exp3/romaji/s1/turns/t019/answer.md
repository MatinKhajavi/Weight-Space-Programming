```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
q,k,_,_=headmats(0,3)
ev,Y=np.linalg.eig(k@q.T)
idx=np.argsort(-np.abs(ev))
planes=[]
for i in idx:
    e=ev[i]
    if e.imag<=1e-9: continue
    x=q.T@Y[:,i]
    B,_=np.linalg.qr(np.stack([x.real,x.imag],1))
    planes.append((e,B))
    if len(planes)>=10: break
print('L0H3 planes:',[f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e,_ in planes])
for l in [0,1,2,3]:
    for h in range(6):
        q,k,_,_=headmats(l,h)
        A=q.T@k
        row=[]
        for e,B in planes:
            M=B.T@A@B
            ee=np.linalg.eigvals(M)
            e1=ee[np.argmax(ee.imag)] if np.any(np.abs(ee.imag)>1e-9) else ee[np.argmax(np.abs(ee))]
            row.append(f'{np.linalg.norm(M):.2f}@{np.degrees(np.angle(e1)):4.0f}')
        print(f'L{l}H{h}: '+' '.join(row))
```