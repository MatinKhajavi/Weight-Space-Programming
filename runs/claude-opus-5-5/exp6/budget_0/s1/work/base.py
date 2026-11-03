
import numpy as np
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
def build_L1(inp):
    W1=np.zeros((243,H),np.float32); B1=np.zeros(H,np.float32)
    for p in pts:
        r,c=p; nb=NB[p]; deg=len(nb)
        W1[inp(r,c,0),U(0,p)]=1
        W1[inp(r,c,1),U(1,p)]=1
        for t,plane,m in ((2,0,1),(3,0,2),(4,1,1),(5,1,2)):
            W1[inp(r,c,plane),U(t,p)]+=m
            for (rr,cc) in nb:
                W1[inp(rr,cc,0),U(t,p)]+=1
                W1[inp(rr,cc,1),U(t,p)]+=1
            B1[U(t,p)]=-deg
    return W1,B1
def rep1(col,sig,p):
    A=V(U(2+2*col,p)); B=V(U(3+2*col,p))
    if (par(p)==0 and sig==0) or (par(p)==1 and sig==1):
        return A
    return B-A
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
