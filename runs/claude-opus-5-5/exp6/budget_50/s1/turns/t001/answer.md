```python
import numpy as np, os, json
print(NAMES)
print({k:SHAPES[k] for k in NAMES})
print(os.listdir('/work'))

builder = r'''
import numpy as np
N=9
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
def nbrs(r,c):
    return [(r+dr,c+dc) for dr,dc in DIRS if 0<=r+dr<N and 0<=c+dc<N]
def P(r,c): return r*9+c
PTS=[(r,c) for r in range(N) for c in range(N)]
def I(r,c,pl): return (r*9+c)*3+pl
H=512
OWN,OPP,SW,SP=0,81,162,243
YW,YP=324,405
A5W,A5P=324,405
AW,AP,SA,EYE=162,243,324,405

def line(r,c): return min(r,8-r,c,8-c)

def build(prm):
    W={}; B={}
    W[1]=np.zeros((243,H),np.float32); B[1]=np.zeros(H,np.float32)
    for k in range(2,7):
        W[k]=np.zeros((H,H),np.float32); B[k]=np.zeros(H,np.float32)
    Wo=np.zeros((H,82),np.float32); bo=np.zeros(82,np.float32)
    # ---- L1
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        W[1][I(r,c,0),OWN+p]=1
        W[1][I(r,c,1),OPP+p]=1
        for blk,pl,bb in ((SW,0,deg-5),(SP,1,deg-5),(YW,0,deg-4),(YP,1,deg-4)):
            W[1][I(r,c,pl),blk+p]=4
            for (rr,cc) in nb:
                W[1][I(rr,cc,0),blk+p]=-1; W[1][I(rr,cc,1),blk+p]=-1
            B[1][blk+p]=bb
    # ---- L2 : S1 with rule (a)+(b)
    K=100.
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        W[2][OWN+p,OWN+p]=1; W[2][OPP+p,OPP+p]=1
        for sblk,yblk,cblk in ((SW,YW,OWN),(SP,YP,OPP)):
            W[2][sblk+p,sblk+p]+=1
            W[2][yblk+p,sblk+p]+=4
            for (rr,cc) in nb:
                q=P(rr,cc)
                W[2][sblk+q,sblk+p]+=4
                W[2][yblk+q,sblk+p]+=1
            W[2][cblk+p,sblk+p]+=K
            B[2][sblk+p]=-4.5-K
    # ---- L3..L5 : propagation
    K2=1e4
    for k in (3,4,5):
        for (r,c) in PTS:
            p=P(r,c); nb=nbrs(r,c)
            W[k][OWN+p,OWN+p]=1; W[k][OPP+p,OPP+p]=1
            for sblk,cblk in ((SW,OWN),(SP,OPP)):
                W[k][sblk+p,sblk+p]+=0.5
                for (rr,cc) in nb:
                    W[k][sblk+P(rr,cc),sblk+p]+=0.5
                W[k][cblk+p,sblk+p]+=K2
                B[k][sblk+p]=-K2
    # L5 extra atari flags from S3
    for (r,c) in PTS:
        p=P(r,c)
        W[5][OWN+p,A5W+p]=1; W[5][SW+p,A5W+p]=-100
        W[5][OPP+p,A5P+p]=1; W[5][SP+p,A5P+p]=-100
    # ---- L6
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        W[6][OWN+p,OWN+p]=1; W[6][OPP+p,OPP+p]=1
        W[6][OWN+p,AW+p]=1; W[6][SW+p,AW+p]=-100
        W[6][OPP+p,AP+p]=1; W[6][SP+p,AP+p]=-100
        for (rr,cc) in nb:
            q=P(rr,cc)
            W[6][OWN+q,SA+p]+=1; W[6][OPP+q,SA+p]+=1
            W[6][SW+q,SA+p]+=-100
            W[6][A5P+q,SA+p]+=-3
            W[6][OWN+q,EYE+p]+=1
            W[6][A5W+q,EYE+p]+=-100
        B[6][SA+p]=2-deg
        B[6][EYE+p]=-(deg-1)
    # ---- output
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[AP+q,p]+=prm['cap']; Wo[AW+q,p]+=prm['esc']
            Wo[OPP+q,p]+=prm['contact']; Wo[OWN+q,p]+=prm['ownadj']
        for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<N and 0<=cc<N:
                Wo[OWN+P(rr,cc),p]+=prm['owndiag']
        Wo[SA+p,p]=-prm['sa']; Wo[EYE+p,p]=-prm['eye']
        bo[p]=prm['lineval'][line(r,c)]
    bo[P(4,4)]+=prm['tengen']
    bo[81]=prm['pass']
    return W,B,Wo,bo

def save_all(save,W,B,Wo,bo):
    for k in range(1,7):
        save('L%d.W'%k,W[k]); save('L%d.b'%k,B[k])
    save('out.W',Wo); save('out.b',bo)
'''
open('/work/builder.py','w').write(builder)
exec(builder)
prm=dict(cap=10.,esc=7.,sa=6.,eye=40.,contact=0.5,ownadj=-0.3,owndiag=0.2,
         lineval=[-1.5,0.0,1.0,0.8,1.0],tengen=0.5,**{'pass':-8.})
json.dump(prm,open('/work/prm_v1.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(saved())
```