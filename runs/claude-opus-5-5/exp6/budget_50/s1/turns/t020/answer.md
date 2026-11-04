```python
builder2 = r'''
import numpy as np
N=9
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
def nbrs(r,c):
    return [(r+dr,c+dc) for dr,dc in DIRS if 0<=r+dr<N and 0<=c+dc<N]
def diags(r,c):
    return [(r+dr,c+dc) for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)) if 0<=r+dr<N and 0<=c+dc<N]
def P(r,c): return r*9+c
PTS=[(r,c) for r in range(N) for c in range(N)]
def I(r,c,pl): return (r*9+c)*3+pl
def line(r,c): return min(r,8-r,c,8-c)
H=512
OWN,OPP,SW,SP=0,81,162,243
YW,YP=324,405
A5W,A5P=324,405
L6_AW,L6_AP,L6_SA,L6_EYE,L6_LIN=0,81,162,243,324

def build(prm):
    W={}; B={}
    W[1]=np.zeros((243,H),np.float32); B[1]=np.zeros(H,np.float32)
    for k in range(2,7):
        W[k]=np.zeros((H,H),np.float32); B[k]=np.zeros(H,np.float32)
    Wo=np.zeros((H,82),np.float32); bo=np.zeros(82,np.float32)
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        W[1][I(r,c,0),OWN+p]=1
        W[1][I(r,c,1),OPP+p]=1
        for blk,pl,bb in ((SW,0,deg-5),(SP,1,deg-5),(YW,0,deg-4),(YP,1,deg-4)):
            W[1][I(r,c,pl),blk+p]=4
            for (rr,cc) in nb:
                W[1][I(rr,cc,0),blk+p]=-1; W[1][I(rr,cc,1),blk+p]=-1
            B[1][blk+p]=bb
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
    # L5 extra: atari flags from S3 (L4) with one readout hop
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        W[5][OWN+p,A5W+p]=1; W[5][OPP+p,A5P+p]=1
        for (rr,cc) in [(r,c)]+nb:
            q=P(rr,cc)
            W[5][SW+q,A5W+p]+=-100; W[5][SP+q,A5P+p]+=-100
    # L6
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
        for (rr,cc) in [(r,c)]+nb:
            q=P(rr,cc)
            W[6][SW+q,L6_AW+p]+=-100; W[6][SP+q,L6_AP+p]+=-100
        for (rr,cc) in nb:
            q=P(rr,cc)
            W[6][OWN+q,L6_SA+p]+=1; W[6][OPP+q,L6_SA+p]+=1
            W[6][SW+q,L6_SA+p]+=-100
            W[6][A5P+q,L6_SA+p]+=-3
            W[6][OWN+q,L6_EYE+p]+=1
            W[6][A5W+q,L6_EYE+p]+=-100
            W[6][OPP+q,L6_LIN+p]+=prm['contact']; W[6][OWN+q,L6_LIN+p]+=prm['ownadj']
        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']
            W[6][OPP+P(rr,cc),L6_LIN+p]+=prm.get('oppdiag',0.)
        B[6][L6_SA+p]=2-deg
        B[6][L6_EYE+p]=-(deg-1)
        B[6][L6_LIN+p]=10.
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']
        Wo[L6_SA+p,p]=-prm['sa']; Wo[L6_EYE+p,p]=-prm['eye']
        Wo[L6_LIN+p,p]=1.
        bo[p]=prm['lineval'][line(r,c)]-10.
    bo[P(4,4)]+=prm['tengen']
    bo[81]=prm['pass']
    return W,B,Wo,bo

def save_all(save,W,B,Wo,bo):
    for k in range(1,7):
        save('L%d.W'%k,W[k]); save('L%d.b'%k,B[k])
    save('out.W',Wo); save('out.b',bo)
'''
open('/work/builder2.py','w').write(builder2)
exec(builder2)
import json
prm=dict(cap=12.,esc=8.,sa=6.,eye=40.,contact=0.6,ownadj=-0.3,owndiag=0.3,oppdiag=0.1,
         lineval=[-2.5,-0.3,1.5,1.2,1.5],tengen=1.0,**{'pass':-8.})
json.dump(prm,open('/work/prm_v2.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print('saved v2')
```