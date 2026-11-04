
import numpy as np
def nb(p):
    r,c=divmod(p,9); out=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(rr*9+cc)
    return out
NB=[nb(p) for p in range(81)]
def dg(p):
    r,c=divmod(p,9)
    return [(r+a)*9+c+d for a,d in ((-1,-1),(-1,1),(1,-1),(1,1)) if 0<=r+a<9 and 0<=c+d<9]
DG=[dg(p) for p in range(81)]
def md(p,q):
    return abs(p//9-q//9)+abs(p%9-q%9)
OWN=0; OPP=81; CONST=162; COLOR=163
A=164; Bq=245; C=326; D=407
def carry(W,b):
    for i in range(162): W[i,i]=1
    b[CONST]=1; W[COLOR,COLOR]=1
def build_trunk():
    Ws={}; bs={}
    W=np.zeros((243,512),np.float32); b=np.zeros(512,np.float32)
    for p in range(81):
        W[p*3+0,OWN+p]=1; W[p*3+1,OPP+p]=1
    b[CONST]=1; W[2,COLOR]=1
    for p in range(81):
        deg=len(NB[p])
        for u,col in ((A+p,0),(Bq+p,1)):
            b[u]=deg-4; W[p*3+col,u]+=4
            for r in NB[p]:
                W[r*3+0,u]-=1; W[r*3+1,u]-=1
    for bi in range(64):
        r0,c0=divmod(bi,8); blk=[(r0+i)*9+c0+j for i in (0,1) for j in (0,1)]
        ext=set(q for s in blk for q in NB[s] if q not in blk)
        for u,col in ((326+bi,0),(390+bi,1)):
            b[u]=len(ext)-1-32
            for s in blk: W[s*3+col,u]+=8
            for x in ext: W[x*3+0,u]-=1; W[x*3+1,u]-=1
    Ws['L1']=W; bs['L1']=b
    BLK=[[bi for bi in range(64) if p in [(bi//8+i)*9+bi%8+j for i in (0,1) for j in (0,1)]] for p in range(81)]
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32); carry(W,b)
    for p in range(81):
        for (src,gate,dst,th) in ((A,OWN,A,2),(Bq,OPP,Bq,2),(A,OWN,C,4),(Bq,OPP,D,4)):
            u=dst+p
            W[src+p,u]+=2
            for r in NB[p]: W[src+r,u]+=1
            W[gate+p,u]+=20; b[u]=-th-20
            if th==2:
                for bi in BLK[p]: W[(326 if src==A else 390)+bi,u]+=3
    Ws['L2']=W; bs['L2']=b
    for L,Bg in (('L3',2048),('L4',16384)):
        W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32); carry(W,b)
        for p in range(81):
            for (ch,gate) in ((A,OWN),(Bq,OPP),(C,OWN),(D,OPP)):
                u=ch+p; W[ch+p,u]+=1
                for r in NB[p]: W[ch+r,u]+=1
                W[gate+p,u]+=Bg; b[u]=-Bg
        Ws[L]=W; bs[L]=b
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32); carry(W,b)
    for p in range(81):
        for (ch,gate) in ((A,OWN),(Bq,OPP),(C,OWN),(D,OPP)):
            u=ch+p; W[gate+p,u]+=1; W[ch+p,u]-=1
            for r in NB[p]: W[ch+r,u]-=1
    Ws['L5']=W; bs['L5']=b
    return Ws,bs
AO=A; AX=Bq; LO=C; LX=D
LIN=0;SAVE=81;SELF=162;ATT=243;EYE=324;INT=405;C6=486
def prior_of(P):
    pr=np.zeros(81,np.float32)
    for p in range(81):
        r,c=divmod(p,9); d=min(r,c,8-r,8-c)
        pr[p]=P['prior'][d]
    pr[40]+=P['center_bonus']
    return pr
def build_head(P):
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    b[C6]=1
    for p in range(81):
        nbp=NB[p]; deg=len(nbp)
        def add_e(u,w):
            b[u]+=w*deg
            for q in nbp: W[OWN+q,u]-=w; W[OPP+q,u]-=w
        u=LIN+p; b[u]+=P['K']; add_e(u,P['w_e'])
        for q in nbp:
            W[AX+q,u]+=P['w_cap']; W[OPP+q,u]+=P['w_oppadj']; W[OWN+q,u]+=P['w_ownadj']
        for q in DG[p]:
            W[OPP+q,u]+=P['w_oppdiag']; W[OWN+q,u]+=P['w_owndiag']
        u=SAVE+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[AO+q,u]+=4; W[OWN+q,u]+=1; W[LO+q,u]-=1
        u=SELF+p; b[u]+=2; add_e(u,-1)
        for q in nbp: W[OWN+q,u]-=2; W[LO+q,u]+=2; W[AX+q,u]-=3
        u=ATT+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[AX+q,u]-=4; W[OWN+q,u]+=1; W[LO+q,u]-=1
        u=EYE+p; b[u]+=1-deg
        for q in nbp: W[OWN+q,u]+=1
        u=INT+p; b[u]-=P['int_T']
        for q in nbp: W[OWN+q,u]+=P['int_own']
        for q in range(81):
            W[OWN+q,u]+=P['int_N']; W[OPP+q,u]+=P['int_N']
            if md(p,q)<=2: W[OPP+q,u]-=P['int_M']
    Wo=np.zeros((512,82),np.float32); bo=np.zeros(82,np.float32)
    pr=prior_of(P)
    for p in range(81):
        Wo[LIN+p,p]=1; bo[p]=-P['K']+pr[p]
        Wo[SAVE+p,p]=P['w_save']; Wo[SELF+p,p]=P['w_self']; Wo[ATT+p,p]=P['w_att']
        Wo[EYE+p,p]=P['w_eye']; Wo[INT+p,p]=P['w_int']
    bo[81]=P['pass']
    s=P['scale']
    return W,b,Wo*s,bo*s
P0=dict(K=10,w_e=0.2,w_cap=10,w_oppadj=0.6,w_ownadj=0.1,w_oppdiag=0.2,w_owndiag=0.4,
        w_save=8,w_self=-6,w_att=2,w_eye=-30,w_int=-3,int_T=2,int_own=1,int_N=0.05,int_M=10,
        prior=[-1.5,0,0.8,0.8,1.0],center_bonus=1.0,scale=1.5)
P0['pass']=-3
def build_all(P):
    Ws,bs=build_trunk()
    W6,b6,Wo,bo=build_head(P)
    Ws['L6']=W6; bs['L6']=b6; Ws['out']=Wo; bs['out']=bo
    return Ws,bs
def save_all(P):
    Ws,bs=build_all(P)
    for k in ['L1','L2','L3','L4','L5','L6','out']:
        save(k+'.W',Ws[k]); save(k+'.b',bs[k])
