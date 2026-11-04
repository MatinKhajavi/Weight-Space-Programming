```python
code = r'''
import numpy as np
B = 30.0
def NB(p):
    r,c = divmod(p,9); o=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc = r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: o.append(rr*9+cc)
    return o
NBS = [NB(p) for p in range(81)]
PAR = [ (p//9 + p%9)%2 for p in range(81)]
LINE = [ min(p//9, p%9, 8-p//9, 8-p%9) for p in range(81)]
class Layer:
    def __init__(self, nin, nout=512):
        self.W = np.zeros((nin,nout),np.float32); self.b = np.zeros(nout,np.float32); self.n=0
    def alloc(self,k):
        idx = list(range(self.n,self.n+k)); self.n += k
        assert self.n <= self.W.shape[1], self.n
        return idx
def kval(dr,dc,kern):
    a,b = sorted((abs(dr),abs(dc)))
    return kern.get((a,b),0.0)
def build(P):
    L1 = Layer(243)
    o1=L1.alloc(81); x1=L1.alloc(81)
    u={'o':L1.alloc(81),'x':L1.alloc(81)}; e1={'o':L1.alloc(81),'x':L1.alloc(81)}
    for p in range(81):
        L1.W[3*p+0,o1[p]]=1; L1.W[3*p+1,x1[p]]=1
        deg=len(NBS[p])
        for cn,col in (('o',0),('x',1)):
            for U,base in ((u[cn],0.0),(e1[cn],-1.0)):
                for q in NBS[p]:
                    L1.W[3*q+0,U[p]]-=1; L1.W[3*q+1,U[p]]-=1
                L1.W[3*p+col,U[p]]+=4
                L1.b[U[p]] = deg-4+base
    L2 = Layer(512)
    o2=L2.alloc(81); x2=L2.alloc(81)
    keys=[('o',0),('o',1),('x',0),('x',1)]
    NP={k:L2.alloc(81) for k in keys}
    for p in range(81):
        L2.W[o1[p],o2[p]]=1; L2.W[x1[p],x2[p]]=1
    for (cn,sg),U in NP.items():
        c1 = o1 if cn=='o' else x1
        for s in range(81):
            L2.W[c1[s],U[s]] += 1
            for xx in [s]+NBS[s]:
                if PAR[xx]==sg: L2.W[u[cn][xx],U[s]] -= 1
                L2.W[e1[cn][xx],U[s]] -= 1
    layers=[L1,L2]
    prev=dict(o=o2,x=x2,NP=NP)
    for li in range(2):
        L = Layer(512)
        cur=dict(o=L.alloc(81),x=L.alloc(81))
        cur['NP']={k:L.alloc(81) for k in keys}
        for p in range(81):
            L.W[prev['o'][p],cur['o'][p]]=1; L.W[prev['x'][p],cur['x'][p]]=1
        for k in keys:
            Up=prev['NP'][k]; Un=cur['NP'][k]; c=prev[k[0]]
            for s in range(81):
                L.W[Up[s],Un[s]]+=1
                for t in NBS[s]:
                    L.W[c[t],Un[s]]-=1; L.W[Up[t],Un[s]]+=1
        layers.append(L); prev=cur
    # L5
    L5 = Layer(512)
    o4,x4,N4 = prev['o'],prev['x'],prev['NP']
    o5=L5.alloc(81); x5=L5.alloc(81); NPo=L5.alloc(81); sx=L5.alloc(81); mx=L5.alloc(81); a2=L5.alloc(81)
    for q in range(81):
        L5.W[o4[q],o5[q]]=1; L5.W[x4[q],x5[q]]=1
        sg=1-PAR[q]; deg=len(NBS[q])
        Uo=N4[('o',sg)]; Ux=N4[('x',sg)]
        L5.W[Uo[q],NPo[q]]+=1
        for t in NBS[q]:
            L5.W[o4[t],NPo[q]]-=1; L5.W[Uo[t],NPo[q]]+=1
        # sx = relu(NPx(q) - sum_t (2x(t) - NPx(t)))
        L5.W[Ux[q],sx[q]]+=1
        for t in NBS[q]:
            L5.W[x4[t],sx[q]]-=2; L5.W[Ux[t],sx[q]]+=1
        # mx = relu(4NPx(q) + sum_t(-3x(t)+4NPx(t)) - 4)
        L5.W[Ux[q],mx[q]]+=4
        for t in NBS[q]:
            L5.W[x4[t],mx[q]]-=3; L5.W[Ux[t],mx[q]]+=4
        L5.b[mx[q]]=-4
        # a2 = relu(3x(q) - l(q) - 3 sum_t x(t)), l = deg - sum(o+x)
        L5.W[x4[q],a2[q]]+=3
        for t in NBS[q]:
            L5.W[o4[t],a2[q]]+=1; L5.W[x4[t],a2[q]]+=1-3
        L5.b[a2[q]]=-deg
    layers.append(L5)
    # L6
    L6 = Layer(512)
    eye6=L6.alloc(81); SA6=L6.alloc(81); Lin6=L6.alloc(81); t1=L6.alloc(81); t2=L6.alloc(81); t3=L6.alloc(81); E6=L6.alloc(1)[0]
    C=P['C']; al=P['alpha']
    for p in range(81):
        deg=len(NBS[p])
        for q in NBS[p]: L6.W[o5[q],eye6[p]]+=1
        L6.W[o5[p],eye6[p]]-=B; L6.W[x5[p],eye6[p]]-=B
        L6.b[eye6[p]] = -(deg-1)
        for q in NBS[p]:
            L6.W[x5[q],SA6[p]] += 1
            L6.W[NPo[q],SA6[p]] += 1
            L6.W[sx[q],SA6[p]] -= 2
        L6.W[o5[p],SA6[p]]-=B; L6.W[x5[p],SA6[p]]-=B
        L6.b[SA6[p]] = 2-deg
        for q in NBS[p]:
            L6.W[sx[q],Lin6[p]] += P['Wc']
            L6.W[mx[q],Lin6[p]] += P['Wm']
            L6.W[NPo[q],Lin6[p]] += P['We']
            L6.W[a2[q],Lin6[p]] += P['Wa']
        L6.b[Lin6[p]] = C
        r0,c0=divmod(p,9)
        for s in range(81):
            if s==p: continue
            r1,c1=divmod(s,9); k=kval(r1-r0,c1-c0,P['kern'])
            if k:
                kw=k/P['w']
                for t in (t1,t2,t3):
                    L6.W[o5[s],t[p]] += kw; L6.W[x5[s],t[p]] -= kw
                    L6.W[NPo[s],t[p]] -= al*kw
                    L6.W[sx[s],t[p]] += al*kw
                    L6.W[mx[s],t[p]] += 0.5*al*kw
        for t,off in ((t1,1),(t2,0),(t3,-1)):
            L6.W[o5[p],t[p]] -= B; L6.W[x5[p],t[p]] -= B
            L6.b[t[p]] = off
    for s in range(81):
        L6.W[o5[s],E6] -= 1; L6.W[x5[s],E6] -= 1
    L6.b[E6] = 81 - P['Ephase']
    layers.append(L6)
    OW=np.zeros((512,82),np.float32); Ob=np.zeros(82,np.float32)
    kp=dict(P['kern']); kp[(0,0)]=P['k0']
    for p in range(81):
        OW[Lin6[p],p]+=1; Ob[p]-=C
        OW[eye6[p],p]-=P['Weye']; OW[SA6[p],p]-=P['Wsa']
        r0,c0=divmod(p,9)
        for r in range(81):
            r1,c1=divmod(r,9); k=kval(r1-r0,c1-c0,kp)
            if k:
                OW[t1[r],p]+=P['Wt']*k; OW[t2[r],p]-=2*P['Wt']*k; OW[t3[r],p]+=P['Wt']*k
        Ob[p]+=P['prior'][LINE[p]]
        OW[E6,p] -= P['edgeph'][LINE[p]]
    Ob[81]=P['P0']
    return layers,OW,Ob
DEFAULT = dict(C=1.0, Wc=14.0, Wm=8.0, We=9.0, Wa=3.0, Weye=40.0, Wsa=25.0, Wt=1.0, w=2.0, k0=1.0, P0=0.5,
    alpha=0.6, Ephase=30.0, edgeph=[0.06,0.02,0.0,0.0,0.0],
    kern={(0,1):1.0,(1,1):0.6,(0,2):0.4,(1,2):0.25,(2,2):0.1,(0,3):0.15,(1,3):0.07},
    prior=[-0.5,-0.2,0.0,0.1,0.3])
def save_all(layers, OW, Ob, save):
    for i,L in enumerate(layers):
        save('L%d.W'%(i+1), L.W); save('L%d.b'%(i+1), L.b)
    save('out.W', OW); save('out.b', Ob)
'''
open('/work/build4.py','w').write(code)
exec(code)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers])
```