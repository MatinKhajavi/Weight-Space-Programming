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
def dist2pts(p):
    r,c = divmod(p,9); o=[]
    for rr in range(9):
        for cc in range(9):
            if abs(rr-r)+abs(cc-c)==2: o.append(rr*9+cc)
    return o
D2 = [dist2pts(p) for p in range(81)]
class Layer:
    def __init__(self, nin, nout=512):
        self.W = np.zeros((nin,nout),np.float32); self.b = np.zeros(nout,np.float32); self.n=0
    def alloc(self,k):
        idx = list(range(self.n,self.n+k)); self.n += k
        assert self.n <= self.W.shape[1], self.n
        return idx
def kern(dr,dc,P):
    a,b = sorted((abs(dr),abs(dc)))
    return P['kern'].get((a,b),0.0)
def build(P):
    L1 = Layer(243)
    o1=L1.alloc(81); x1=L1.alloc(81); pl1=L1.alloc(1)[0]; uo1=L1.alloc(81); ux1=L1.alloc(81)
    for p in range(81):
        L1.W[3*p+0,o1[p]]=1; L1.W[3*p+1,x1[p]]=1
        for u,col in ((uo1,0),(ux1,1)):
            for q in NBS[p]:
                L1.W[3*q+0,u[p]]-=1; L1.W[3*q+1,u[p]]-=1
            L1.W[3*p+col,u[p]]+=4
            L1.b[u[p]] = len(NBS[p])-4
    L1.W[2,pl1]=1
    L2 = Layer(512)
    o2=L2.alloc(81); x2=L2.alloc(81); pl2=L2.alloc(1)[0]
    W2o=L2.alloc(81); W3o=L2.alloc(81); W2x=L2.alloc(81); W3x=L2.alloc(81)
    for p in range(81):
        L2.W[o1[p],o2[p]]=1; L2.W[x1[p],x2[p]]=1
    L2.W[pl1,pl2]=1
    for s in range(81):
        for Wk,u,c,thr in ((W2o,uo1,o1,1),(W3o,uo1,o1,2),(W2x,ux1,x1,1),(W3x,ux1,x1,2)):
            L2.W[u[s],Wk[s]]+=1
            for t in NBS[s]: L2.W[u[t],Wk[s]]+=1
            L2.W[c[s],Wk[s]]+=B
            L2.b[Wk[s]] = -thr-B
    L3 = Layer(512)
    o3=L3.alloc(81); x3=L3.alloc(81); pl3=L3.alloc(1)[0]
    U2o3=L3.alloc(81); U3o3=L3.alloc(81); U2x3=L3.alloc(81); U3x3=L3.alloc(81)
    for p in range(81):
        L3.W[o2[p],o3[p]]=1; L3.W[x2[p],x3[p]]=1
    L3.W[pl2,pl3]=1
    for s in range(81):
        for U,Wk,c in ((U2o3,W2o,o2),(U3o3,W3o,o2),(U2x3,W2x,x2),(U3x3,W3x,x2)):
            L3.W[c[s],U[s]]+=1
            L3.W[Wk[s],U[s]]-=1
            for t in NBS[s]: L3.W[Wk[t],U[s]]-=1
    prev = dict(o=o3,x=x3,pl=pl3,U2o=U2o3,U3o=U3o3,U2x=U2x3,U3x=U3x3)
    layers=[L1,L2,L3]
    for li in range(2):
        L = Layer(512)
        cur = dict(o=L.alloc(81), x=L.alloc(81), pl=L.alloc(1)[0], U2o=L.alloc(81),U3o=L.alloc(81),U2x=L.alloc(81),U3x=L.alloc(81))
        for p in range(81):
            L.W[prev['o'][p],cur['o'][p]]=1; L.W[prev['x'][p],cur['x'][p]]=1
        L.W[prev['pl'],cur['pl']]=1
        for name,cn in (('U2o','o'),('U3o','o'),('U2x','x'),('U3x','x')):
            U=prev[name]; c=prev[cn]; Un=cur[name]
            for s in range(81):
                L.W[U[s],Un[s]]+=1
                for t in NBS[s]:
                    L.W[c[t],Un[s]]-=1; L.W[U[t],Un[s]]+=1
        layers.append(L); prev=cur
    L6 = Layer(512)
    o5,x5 = prev['o'],prev['x']; U2o5,U3o5,U2x5,U3x5 = prev['U2o'],prev['U3o'],prev['U2x'],prev['U3x']
    eye6=L6.alloc(81); SA6=L6.alloc(81); Lin6=L6.alloc(81); t1=L6.alloc(81); t2=L6.alloc(81); t3=L6.alloc(81)
    C = P['C']
    for p in range(81):
        deg=len(NBS[p])
        for q in NBS[p]: L6.W[o5[q],eye6[p]]+=1
        L6.W[o5[p],eye6[p]]-=B; L6.W[x5[p],eye6[p]]-=B
        L6.b[eye6[p]] = -(deg-1)
        for q in NBS[p]:
            L6.W[o5[q],SA6[p]]-=1; L6.W[x5[q],SA6[p]]+=1
            L6.W[U2o5[q],SA6[p]]+=1; L6.W[U3o5[q],SA6[p]]+=1
            L6.W[U2x5[q],SA6[p]]-=2
        L6.W[o5[p],SA6[p]]-=B; L6.W[x5[p],SA6[p]]-=B
        L6.b[SA6[p]] = 2-deg
        for q in NBS[p]:
            L6.W[U2x5[q],Lin6[p]] += P['Wc'] - P['Wa']
            L6.W[U3x5[q],Lin6[p]] += P['Wa']
            L6.W[U2o5[q],Lin6[p]] += P['We'] - P['Wd']
            L6.W[U3o5[q],Lin6[p]] += P['Wd']
            L6.W[o5[q],Lin6[p]] -= P['Wlib']; L6.W[x5[q],Lin6[p]] -= P['Wlib']
        for q in D2[p]:
            L6.W[U2x5[q],Lin6[p]] += P['Wc2']
            L6.W[U2o5[q],Lin6[p]] += P['We2']
        L6.b[Lin6[p]] = C + P['Wlib']*deg
        r0,c0 = divmod(p,9)
        for s in range(81):
            if s==p: continue
            r1,c1 = divmod(s,9)
            k = kern(r1-r0,c1-c0,P)
            if k:
                for t in (t1,t2,t3):
                    L6.W[o5[s],t[p]] += k/P['w']; L6.W[x5[s],t[p]] -= k/P['w']
        for t,off in ((t1,1),(t2,0),(t3,-1)):
            L6.W[o5[p],t[p]] -= B; L6.W[x5[p],t[p]] -= B
            L6.b[t[p]] = off
    layers.append(L6)
    OW = np.zeros((512,82),np.float32); Ob = np.zeros(82,np.float32)
    kp = dict(P['kern']); kp[(0,0)] = P['k0']
    for p in range(81):
        OW[Lin6[p],p] += 1; Ob[p] -= C
        OW[eye6[p],p] -= P['Weye']
        OW[SA6[p],p] -= P['Wsa']
        r0,c0 = divmod(p,9)
        for r in range(81):
            r1,c1 = divmod(r,9)
            a,b2 = sorted((abs(r1-r0),abs(c1-c0)))
            k = kp.get((a,b2),0.0)
            if k:
                OW[t1[r],p] += P['Wt']*k; OW[t2[r],p] -= 2*P['Wt']*k; OW[t3[r],p] += P['Wt']*k
        line = min(r0,c0,8-r0,8-c0)
        Ob[p] += P['prior'][line]
    Ob[81] = P['P0']
    return layers, OW, Ob
DEFAULT = dict(C=1.0, Wc=8.0, Wa=2.0, We=6.0, Wd=1.5, Wlib=0.0, Wc2=1.5, We2=1.0,
               Weye=40.0, Wsa=10.0, Wt=1.0, w=2.0, k0=1.0, P0=0.5,
               kern={(0,1):1.0,(1,1):0.6,(0,2):0.4,(1,2):0.25,(2,2):0.1,(0,3):0.15,(1,3):0.07},
               prior=[-0.5,-0.2,0.0,0.1,0.3])
def save_all(layers, OW, Ob, save):
    for i,L in enumerate(layers):
        save('L%d.W'%(i+1), L.W); save('L%d.b'%(i+1), L.b)
    save('out.W', OW); save('out.b', Ob)
'''
open('/work/build.py','w').write(code)
exec(code)
layers, OW, Ob = build(DEFAULT)
print([L.n for L in layers])
save_all(layers, OW, Ob, save)
print(saved())

import json
def B2(rows, tm='black'): return {"board": rows, "to_move": tm}
E = '.........'
pos = []
pos.append(B2([E]*9))
pos.append(B2([E,E,'...X.....','..XO.....','...X.....',E,E,E,E]))
pos.append(B2([E,'...O.....','..OX.....','...O.....',E,E,E,E,E]))
pos.append(B2([E,E,'...XX....','..XOO....','...XX....',E,E,E,E]))
pos.append(B2(['OOOOOO...','XXXXXX...',E,E,E,E,E,E,E]))
pos.append(B2([E,'...X.....','..XOX....','..XOO....','...XX....',E,E,E,E]))
pos.append(B2(['X.XXO....','XXXXO....','OOOOO....',E,E,E,E,E,E]))
pos.append(B2(['.X.O.....','XO.O.....','.OOO.....',E,E,E,E,E,E]))
pos.append(B2([E,E,E,E,'....X....',E,E,E,E],'white'))
pos.append(B2([E,E,E,'..XXX....','.XOOO....','..XXX....',E,E,E]))
pos.append(B2([E,E,E,'..XXX....','.XOOO....','..XXX....',E,E,E],'white'))
pos.append(B2(['...XO....','...XO....','XXXXO....','OOOOO....',E,E,E,E,E]))
req = {"positions": pos, "activations": True}
json.dump(req, open('/work/req1.json','w'))
print(json.dumps(req))
```