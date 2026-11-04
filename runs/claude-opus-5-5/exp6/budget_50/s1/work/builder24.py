
import numpy as np
N=9
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
def nbrs(r,c):
    return [(r+dr,c+dc) for dr,dc in DIRS if 0<=r+dr<N and 0<=c+dc<N]
def diags(r,c):
    return [(r+dr,c+dc) for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)) if 0<=r+dr<N and 0<=c+dc<N]
def ring2(r,c):
    return [(r+dr,c+dc) for dr in range(-2,3) for dc in range(-2,3) if abs(dr)+abs(dc)==2 and 0<=r+dr<N and 0<=c+dc<N]
def P(r,c): return r*9+c
PTS=[(r,c) for r in range(N) for c in range(N)]
def I(r,c,pl): return (r*9+c)*3+pl
def line(r,c): return min(r,8-r,c,8-c)
def par(r,c): return (r+c)%2
H=512
# L1 blocks
C1,EW,OW,S3W,EP,OP=0,81,162,243,324,405
# L2..L5 blocks
OWN,OPP,SW,S3,XE,XO=0,81,162,243,324,405
# L6 blocks
L6_AW,L6_AP,L6_SA,L6_EYE,L6_LIN,L6_OT=0,81,162,243,324,405

def build(prm):
    W={}; B={}
    W[1]=np.zeros((243,H),np.float32); B[1]=np.zeros(H,np.float32)
    for k in range(2,7):
        W[k]=np.zeros((H,H),np.float32); B[k]=np.zeros(H,np.float32)
    Wo=np.zeros((H,82),np.float32); bo=np.zeros(82,np.float32)
    # ---- L1
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb); even=par(r,c)==0
        W[1][I(r,c,0),C1+p]=1; W[1][I(r,c,1),C1+p]=-1; B[1][C1+p]=1
        # own parity blocks: EW gets Y' at even, S0 at odd ; OW gets S0 at even, Y' at odd
        specs=[(EW,0,deg-4 if even else deg-5),(OW,0,deg-5 if even else deg-4),(S3W,0,deg-6),
               (EP,1,deg-4 if even else deg-5),(OP,1,deg-5 if even else deg-4)]
        for blk,pl,bb in specs:
            W[1][I(r,c,pl),blk+p]=4
            for (rr,cc) in nb:
                W[1][I(rr,cc,0),blk+p]=-1; W[1][I(rr,cc,1),blk+p]=-1
            B[1][blk+p]=bb
    def Yp(r,c): return EW if par(r,c)==0 else OW
    def S0b(r,c): return OW if par(r,c)==0 else EW
    # ---- L2
    K=100.
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        # carry own/opp from C
        W[2][C1+p,OWN+p]=1; B[2][OWN+p]=-1
        W[2][C1+p,OPP+p]=-1; B[2][OPP+p]=1
        # S1: 5*(S0_s+sum S0_t) + 4*I1_s + sum I1_t - 4.5 ; I1=Y'-S0
        ys,ss=Yp(r,c),S0b(r,c)
        W[2][ss+p,SW+p]+=5-4; W[2][ys+p,SW+p]+=4
        for (rr,cc) in nb:
            q=P(rr,cc); yt,st=Yp(rr,cc),S0b(rr,cc)
            W[2][st+q,SW+p]+=5-1; W[2][yt+q,SW+p]+=1
        W[2][C1+p,SW+p]+=K; B[2][SW+p]=-4.5-2*K
        # S3_1: 5*(S30_s+sum S30_t) + 0.5*I2_s + 4*I1_s + sum(0.1 I1_t + I2_t) - 4.5
        # I2=S0-S30
        W[2][S3W+p,S3+p]+=5-0.5; W[2][ss+p,S3+p]+=0.5-4; W[2][ys+p,S3+p]+=4
        for (rr,cc) in nb:
            q=P(rr,cc); yt,st=Yp(rr,cc),S0b(rr,cc)
            W[2][S3W+q,S3+p]+=5-1; W[2][st+q,S3+p]+=-0.1+1; W[2][yt+q,S3+p]+=0.1
        W[2][C1+p,S3+p]+=K; B[2][S3+p]=-4.5-2*K
        # opp parity propagate, mask opp (C==0)
        for src,dst in ((EP,XE),(OP,XO)):
            W[2][src+p,dst+p]+=0.5
            for (rr,cc) in nb: W[2][src+P(rr,cc),dst+p]+=0.5
            W[2][C1+p,dst+p]+=-K
    # ---- L3..L5
    K2=1e4
    for k in (3,4,5):
        for (r,c) in PTS:
            p=P(r,c); nb=nbrs(r,c)
            W[k][OWN+p,OWN+p]=1; W[k][OPP+p,OPP+p]=1
            for blk,cblk in ((SW,OWN),(S3,OWN),(XE,OPP),(XO,OPP)):
                W[k][blk+p,blk+p]+=0.5
                for (rr,cc) in nb: W[k][blk+P(rr,cc),blk+p]+=0.5
                W[k][cblk+p,blk+p]+=K2; B[k][blk+p]=-K2
    # ---- L6
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        tch = XO if par(r,c)==0 else XE
        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
        for (rr,cc) in PTS:
            dd=abs(rr-r)+abs(cc-c)
            if 0<dd<=2: W[6][OPP+P(rr,cc),L6_AP+p]+=-prm['dens_b']
        W[6][OPP+p,L6_AP+p]=3.
        B[6][L6_AP+p]=prm['dens_b']*prm['dens_t']-2.
        for (rr,cc) in [(r,c)]+nb:
            q=P(rr,cc)
            W[6][SW+q,L6_AW+p]+=-100; W[6][tch+q,L6_AP+p]+=-100
        for (rr,cc) in nb:
            q=P(rr,cc)
            W[6][OWN+q,L6_SA+p]+=1; W[6][OPP+q,L6_SA+p]+=1
            W[6][S3+q,L6_SA+p]+=-100
            W[6][OWN+q,L6_EYE+p]+=prm['kM']*1; W[6][OPP+q,L6_EYE+p]+=-prm['kM']*4
            W[6][OPP+q,L6_LIN+p]+=prm['contact']; W[6][OWN+q,L6_LIN+p]+=prm['ownadj']
            W[6][OWN+q,L6_OT+p]+=1.; W[6][OPP+q,L6_OT+p]+=-2.
        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']
            W[6][OPP+P(rr,cc),L6_LIN+p]+=prm['oppdiag']
        for (rr,cc) in ring2(r,c):
            W[6][OWN+P(rr,cc),L6_OT+p]+=0.5; W[6][OPP+P(rr,cc),L6_OT+p]+=-1.
        B[6][L6_SA+p]=2-deg
        W[6][OPP+p,L6_EYE+p]+=prm['kM']*4
        B[6][L6_EYE+p]=prm['kM']*((4-deg)-7)-7.5-prm['kT']
        for (rr2,cc2) in PTS:
            W[6][OWN+P(rr2,cc2),L6_EYE+p]+=1; W[6][OPP+P(rr2,cc2),L6_EYE+p]+=-1
        W[6][486,L6_EYE+p]+=15.
        B[6][L6_LIN+p]=10.
        B[6][L6_OT+p]=-prm['ot_t']+(4-deg)*prm['ot_deg']
    u=486
    for (cr,cc_) in ((0,0),(0,8),(8,0),(8,8)):
        dr=1 if cr==0 else -1; dc=1 if cc_==0 else -1
        for d1,d2 in (((dr,0),(0,dc)),((0,dc),(dr,0))):
            Cp=(cr,cc_); Bp=(cr+d1[0],cc_+d1[1]); Dp=(cr+d2[0],cc_+d2[1]); Ap=(cr+2*d1[0],cc_+2*d1[1])
            X1=(Bp[0]+d2[0],Bp[1]+d2[1]); X2=(cr+3*d1[0],cc_+3*d1[1]); X3=(Ap[0]+d2[0],Ap[1]+d2[1])
            W[6][OWN+P(*Bp),u]+=1; W[6][OWN+P(*Dp),u]+=1
            for X in (X1,X2,X3): W[6][OPP+P(*X),u]+=1
            W[6][OWN+P(*Cp),u]+=-1; W[6][OPP+P(*Cp),u]+=-1
            B[6][u]=1-5
            Wo[u,P(*Ap)]+=-prm['cyc']
            u+=1
    W[1][I(0,0,2),486]=1.
    for k in (2,3,4,5): W[k][486,486]=1.
    LX=487
    for (r,c) in PTS:
        p=P(r,c); W[5][OWN+p,LX]+=1.; W[5][OPP+p,LX]+=-1.
    W[5][486,LX]+=15.; B[5][LX]=-7.5-prm['TL']
    for (r,c) in PTS:
        W[6][LX,L6_AP+P(r,c)]+=-prm['cl']
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_EYE+q,p]+=-prm['kw']
            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']
        Wo[L6_SA+p,p]=-prm['sa']; pass
        Wo[L6_LIN+p,p]=1.; Wo[L6_OT+p,p]=-prm['ot']
        bo[p]=prm['lineval'][line(r,c)]-10.
    bo[P(4,4)]+=prm['tengen']
    bo[81]=prm['pass']
    return W,B,Wo,bo

def save_all(save,W,B,Wo,bo):
    for k in range(1,7):
        save('L%d.W'%k,W[k]); save('L%d.b'%k,B[k])
    save('out.W',Wo); save('out.b',bo)
