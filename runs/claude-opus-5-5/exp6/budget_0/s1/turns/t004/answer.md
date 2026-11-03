```python
code6 = r'''
def build_L6_out(params=None):
    W6=np.zeros((H,H),np.float32); B6=np.zeros(H,np.float32)
    def set6(tgt,ex):
        for k,v in ex.d.items(): W6[k,tgt]+=v
        B6[tgt]+=ex.c
    K=1.0
    for p in pts:
        nb=NB[p]; deg=len(nb)
        nO=Esum(V(U(0,q)) for q in nb); nX=Esum(V(U(1,q)) for q in nb)
        a=Esum(V(U(2,q))+V(U(3,q)) for q in nb)
        cap=Esum(V(U(4,q))+V(U(5,q)) for q in nb)
        n=deg-nO-nX; s=nO-a; t=nX-cap
        I=E(); emp=E(); nX2=E()
        for q in pts:
            d=dist(p,q)
            if 1<=d<=3:
                I=I+{1:1.0,2:0.5,3:0.25}[d]*(V(U(0,q))-V(U(1,q)))
            if 1<=d<=2:
                emp=emp+(1-V(U(0,q))-V(U(1,q)))
                nX2=nX2+V(U(1,q))
        set6(U(0,p), K+10*cap+11*a+0.7*n+0.5*t+0.1*emp)
        set6(U(1,p), 0.25+a-4*(n-1)-10*(s+cap))
        set6(U(2,p), 2-n-2*(t+cap+a))
        set6(U(3,p), I-0.75-3*nX-3*a)
        set6(U(4,p), -1*I-2.5)
        set6(U(5,p), nX2+12*cap-12)
    Wo=np.zeros((H,82),np.float32); Bo=np.zeros(82,np.float32)
    ring={0:-2.5,1:-1.0,2:0.3,3:0.8,4:1.5}
    for p in pts:
        r,c=p; d=min(r,c,8-r,8-c)
        for t,w in ((0,1.0),(1,-13.0),(2,-15.0),(3,-2.0),(4,-1.0),(5,0.5)):
            Wo[U(t,p),P(p)]=w
        Bo[P(p)]=-K+ring[d]
    Bo[81]=-4.0
    return W6,B6,Wo,Bo
'''
open('/work/l6.py','w').write(code6)
exec(open('/work/base.py').read()); exec(open('/work/l6.py').read())
W1,B1=build_L1(inp)
W2,B2=prop(rep1); W3,B3=prop(repD)
W6,B6,Wo,Bo=build_L6_out()
def show_col(Wm,Bm,tgt,names):
    nz=np.nonzero(Wm[:,tgt])[0]
    items=[]
    for k in nz:
        t,pp=divmod(int(k),81)
        items.append(f"{names[t]}{divmod(pp,9)}:{Wm[k,tgt]:+g}")
    print(" ".join(items), "| bias", Bm[tgt])
n1=['O','X','aO','bO','aX','bX']; nk=['O','X','WxO','WyO','WxX','WyX']
p=(4,4); q=(4,5)
print("L2 WxO at (4,4) even:"); show_col(W2,B2,U(2,p),n1)
print("L2 WyO at (4,4) even:"); show_col(W2,B2,U(3,p),n1)
print("L2 WxX at (4,5) odd:"); show_col(W2,B2,U(4,q),n1)
print("L3 WyX at (0,0):"); show_col(W3,B3,U(5,(0,0)),nk)
print("L6 Lin (4,4):"); show_col(W6,B6,U(0,p),nk)
print("L6 SA (4,4):"); show_col(W6,B6,U(1,p),nk)
print("L6 E2 (0,0):"); show_col(W6,B6,U(2,(0,0)),nk)
print("L6 Cap (0,0):"); show_col(W6,B6,U(5,(0,0)),nk)
print("L1 aO (0,0):", [ (int(i), float(W1[i,U(2,(0,0))])) for i in np.nonzero(W1[:,U(2,(0,0))])[0]], B1[U(2,(0,0))])
print("L1 bX (0,0):", [ (int(i), float(W1[i,U(5,(0,0))])) for i in np.nonzero(W1[:,U(5,(0,0))])[0]], B1[U(5,(0,0))])
```