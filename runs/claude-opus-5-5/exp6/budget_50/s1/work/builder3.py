
import numpy as np
N=9
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
def nbrs(r,c):
    return [(r+dr,c+dc) for dr,dc in DIRS if 0<=r+dr<N and 0<=c+dc<N]
def diags(r,c):
    return [(r+dr,c+dc) for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)) if 0<=r+dr<N and 0<=c+dc<N]
def ring2(r,c):
    out=[]
    for dr in range(-2,3):
        for dc in range(-2,3):
            if abs(dr)+abs(dc)==2 and 0<=r+dr<N and 0<=c+dc<N: out.append((r+dr,c+dc))
    return out
def P(r,c): return r*9+c
PTS=[(r,c) for r in range(N) for c in range(N)]
def I(r,c,pl): return (r*9+c)*3+pl
def line(r,c): return min(r,8-r,c,8-c)
def par(r,c): return (r+c)%2
H=512
OWN,OPP,SW,YW,EP,OP=0,81,162,243,324,405
A5P=243
L6_AW,L6_AP,L6_SA,L6_EYE,L6_LIN,L6_OT=0,81,162,243,324,405

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
        even = par(r,c)==0
        specs=[(SW,0,deg-5),(YW,0,deg-4),
               (EP,1,deg-4 if even else deg-5),(OP,1,deg-5 if even else deg-4)]
        for blk,pl,bb in specs:
            W[1][I(r,c,pl),blk+p]=4
            for (rr,cc) in nb:
                W[1][I(rr,cc,0),blk+p]=-1; W[1][I(rr,cc,1),blk+p]=-1
            B[1][blk+p]=bb
    # ---- L2
    K=100.
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        W[2][OWN+p,OWN+p]=1; W[2][OPP+p,OPP+p]=1
        W[2][SW+p,SW+p]+=1; W[2][YW+p,SW+p]+=4
        for (rr,cc) in nb:
            q=P(rr,cc); W[2][SW+q,SW+p]+=4; W[2][YW+q,SW+p]+=1
        W[2][OWN+p,SW+p]+=K; B[2][SW+p]=-4.5-K
        for blk in (EP,OP):
            W[2][blk+p,blk+p]+=0.5
            for (rr,cc) in nb: W[2][blk+P(rr,cc),blk+p]+=0.5
            W[2][OPP+p,blk+p]+=K; B[2][blk+p]=-K
    # ---- L3..L5
    K2=1e4
    for k in (3,4,5):
        for (r,c) in PTS:
            p=P(r,c); nb=nbrs(r,c)
            W[k][OWN+p,OWN+p]=1; W[k][OPP+p,OPP+p]=1
            for blk,cblk in ((SW,OWN),(EP,OPP),(OP,OPP)):
                W[k][blk+p,blk+p]+=0.5
                for (rr,cc) in nb: W[k][blk+P(rr,cc),blk+p]+=0.5
                W[k][cblk+p,blk+p]+=K2; B[k][blk+p]=-K2
    # L5 opp atari flag (from L4 parity channels + readout hop)
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        tch = OP if par(r,c)==0 else EP
        W[5][OPP+p,A5P+p]=1
        for (rr,cc) in [(r,c)]+nb: W[5][tch+P(rr,cc),A5P+p]+=-100
    # ---- L6
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        tch = OP if par(r,c)==0 else EP
        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
        for (rr,cc) in [(r,c)]+nb:
            q=P(rr,cc)
            W[6][SW+q,L6_AW+p]+=-100; W[6][tch+q,L6_AP+p]+=-100
        for (rr,cc) in nb:
            q=P(rr,cc)
            W[6][OWN+q,L6_SA+p]+=1; W[6][OPP+q,L6_SA+p]+=1
            W[6][SW+q,L6_SA+p]+=-100
            W[6][A5P+q,L6_SA+p]+=-prm['saover']
            W[6][OWN+q,L6_EYE+p]+=1
            W[6][OPP+q,L6_LIN+p]+=prm['contact']; W[6][OWN+q,L6_LIN+p]+=prm['ownadj']
            W[6][OWN+q,L6_OT+p]+=1.; W[6][OPP+q,L6_OT+p]+=-2.
        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']
            W[6][OPP+P(rr,cc),L6_LIN+p]+=prm['oppdiag']
        for (rr,cc) in ring2(r,c):
            W[6][OWN+P(rr,cc),L6_OT+p]+=0.5; W[6][OPP+P(rr,cc),L6_OT+p]+=-1.
        B[6][L6_SA+p]=2-deg
        B[6][L6_EYE+p]=-(deg-1)
        B[6][L6_LIN+p]=10.
        B[6][L6_OT+p]=-prm['ot_t']
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']
        Wo[L6_SA+p,p]=-prm['sa']; Wo[L6_EYE+p,p]=-prm['eye']
        Wo[L6_LIN+p,p]=1.; Wo[L6_OT+p,p]=-prm['ot']
        bo[p]=prm['lineval'][line(r,c)]-10.
    bo[P(4,4)]+=prm['tengen']
    bo[81]=prm['pass']
    return W,B,Wo,bo

def save_all(save,W,B,Wo,bo):
    for k in range(1,7):
        save('L%d.W'%k,W[k]); save('L%d.b'%k,B[k])
    save('out.W',Wo); save('out.b',bo)
