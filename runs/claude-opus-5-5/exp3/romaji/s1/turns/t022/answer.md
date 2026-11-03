```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
sink=[21,39,58,99,184]
q3,k3,v3,o3=headmats(0,3); OV3=o3@v3
Uo,So,Vto=np.linalg.svd(OV3)
T=Vto[:24].T
# positional subspace: top right sing vecs of [q3;k3], minus T
X=np.vstack([q3,k3])
Pz=np.linalg.svd(X)[2][:30].T
Pz=Pz-T@(T.T@Pz)
Pz,_=np.linalg.qr(Pz)
print('overlap check T vs Pz', np.linalg.norm(T.T@Pz))
Pmov=Uo[:,:24]  # L0H3 output subspace (prev token)
Pmov=Pmov-T@(T.T@Pmov); Pmov=Pmov-Pz@(Pz.T@Pmov); Pmov,_=np.linalg.qr(Pmov)
Sk=np.zeros((384,5)); Sk[sink,range(5)]=1
subs={'tok':T,'pos':Pz,'prev':Pmov,'sink':Sk}
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=headmats(l,h)
        A=q.T@k
        tot=np.linalg.norm(A)
        out=[]
        for a in ['tok','pos','prev']:
            for b in ['tok','pos','prev','sink']:
                B=subs[a].T@A@subs[b]
                out.append(f'{a[0]}{b[0]}:{np.linalg.norm(B)/tot:.2f}')
        # pos-pos eigen
        B=Pz.T@A@Pz; ev=np.linalg.eigvals(B); ev=ev[np.argsort(-np.abs(ev))][:4]
        print(f'L{l}H{h} '+' '.join(out)+'  pp-eig '+' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev))
```