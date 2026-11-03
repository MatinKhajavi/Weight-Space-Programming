```python
import numpy as np
print(NAMES)
print(SHAPES)
H=512
pts=[(r,c) for r in range(9) for c in range(9)]
def P(p): return p[0]*9+p[1]
NB={p:[(p[0]+dr,p[1]+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)) if 0<=p[0]+dr<9 and 0<=p[1]+dc<9] for p in pts}
def U(t,p): return t*81+P(p)
def par(p): return (p[0]+p[1])%2
def dist(p,q): return abs(p[0]-q[0])+abs(p[1]-q[1])

class E:
    def __init__(s,d=None,c=0.0): s.d=dict(d or {}); s.c=float(c)
    def __add__(s,o):
        if not isinstance(o,E): o=E(c=o)
        d=dict(s.d)
        for k,v in o.d.items(): d[k]=d.get(k,0.0)+v
        return E(d,s.c+o.c)
    __radd__=__add__
    def __mul__(s,k): return E({a:b*k for a,b in s.d.items()},s.c*k)
    __rmul__=__mul__
    def __neg__(s): return s*(-1)
    def __sub__(s,o): return s+(-o if isinstance(o,E) else E(c=-o))
    def __rsub__(s,o): return (-s)+o
def V(u): return E({u:1.0})
def Esum(l):
    r=E()
    for x in l: r=r+x
    return r

# ---------------- L1 : local features ----------------
W1=np.zeros((243,H),np.float32); B1=np.zeros(H,np.float32)
for p in pts:
    r,c=p; nb=NB[p]; deg=len(nb)
    W1[inp(r,c,0),U(0,p)]=1      # own stone
    W1[inp(r,c,1),U(1,p)]=1      # opp stone
    # a=relu(stone - nE), b=relu(2 stone - nE);  nE = deg - sum_n(occupied)
    for t,plane,m in ((2,0,1),(3,0,2),(4,1,1),(5,1,2)):
        W1[inp(r,c,plane),U(t,p)]+=m
        for (rr,cc) in nb:
            W1[inp(rr,cc,0),U(t,p)]+=1
            W1[inp(rr,cc,1),U(t,p)]+=1
        B1[U(t,p)]=-deg

# "no-signal" bits at layer 1 expressed linearly in L1 units
def rep1(col,sig,p):
    A=V(U(2+2*col,p)); B=V(U(3+2*col,p))
    if (par(p)==0 and sig==0) or (par(p)==1 and sig==1):
        return A          # stone with nE==0
    return B-A            # stone with nE<=1
def repD(col,sig,p): return V(U(2+2*col+sig,p))

def prop(rep):
    Wm=np.zeros((H,H),np.float32); Bm=np.zeros(H,np.float32)
    def setu(tgt,ex):
        for k,v in ex.d.items(): Wm[k,tgt]+=v
        Bm[tgt]+=ex.c
    for p in pts:
        setu(U(0,p),V(U(0,p)))
        setu(U(1,p),V(U(1,p)))
        for col in (0,1):
            for sig in (0,1):
                ex=rep(col,sig,p)+V(U(col,p))-1
                for q in NB[p]:
                    ex=ex+rep(col,sig,q)-V(U(col,q))
                setu(U(2+2*col+sig,p),ex)
    return Wm,Bm

W2,B2=prop(rep1)
W3,B3=prop(repD)
W4,B4=prop(repD)
W5,B5=prop(repD)

# ---------------- L6 : move features ----------------
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
    set6(U(0,p), K+10*cap+11*a+0.7*n+0.5*t+0.1*emp)          # Lin
    set6(U(1,p), 0.25+a-4*(n-1)-10*(s+cap))                   # self-atari
    set6(U(2,p), 2-n-2*(t+cap+a))                             # eye / near-eye
    set6(U(3,p), I-0.75-3*nX-3*a)                             # deep own area
    set6(U(4,p), -1*I-2.5)                                    # deep opp area
    set6(U(5,p), nX2+12*cap-12)                               # capture size proxy

Wo=np.zeros((H,82),np.float32); Bo=np.zeros(82,np.float32)
ring={0:-2.5,1:-1.0,2:0.3,3:0.8,4:1.5}
for p in pts:
    r,c=p; d=min(r,c,8-r,8-c)
    for t,w in ((0,1.0),(1,-13.0),(2,-15.0),(3,-2.0),(4,-1.0),(5,0.5)):
        Wo[U(t,p),P(p)]=w
    Bo[P(p)]=-K+ring[d]
Bo[81]=-4.0

tensors={'L1.W':W1,'L1.b':B1,'L2.W':W2,'L2.b':B2,'L3.W':W3,'L3.b':B3,'L4.W':W4,'L4.b':B4,
         'L5.W':W5,'L5.b':B5,'L6.W':W6,'L6.b':B6,'out.W':Wo,'out.b':Bo}
for k,v in tensors.items():
    print(k, v.shape, float(np.abs(v).max()), int((v!=0).sum()))
```