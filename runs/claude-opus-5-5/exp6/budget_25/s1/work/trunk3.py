
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
def md(p,q): return abs(p//9-q//9)+abs(p%9-q%9)
OWN=0; OPP=81; CONST=162; COLOR=163
EO=164; EX=245; BO=326; BX=390          # L1
SO=164; SX=245; TX=326; VX=407          # L2-L4 (VX from L3)
AO=164; AX=245; LX=326; AXS=407         # L5
BLKS=[[(r0+i)*9+c0+j for i in (0,1) for j in (0,1)] for r0 in range(8) for c0 in range(8)]
BLK=[[bi for bi in range(64) if p in BLKS[bi]] for p in range(81)]
def carry(W,b):
    for i in range(162): W[i,i]=1
    b[CONST]=1; W[COLOR,COLOR]=1
def Z(n=512): return np.zeros((n,512),np.float32), np.zeros(512,np.float32)
def build_trunk(M=16):
    Ws={}; bs={}
    W,b=Z(243)
    for p in range(81): W[p*3,OWN+p]=1; W[p*3+1,OPP+p]=1
    b[CONST]=1; W[2,COLOR]=1
    for p in range(81):
        deg=len(NB[p])
        for u,col in ((EO+p,0),(EX+p,1)):
            b[u]=deg-4; W[p*3+col,u]+=4
            for r in NB[p]: W[r*3,u]-=1; W[r*3+1,u]-=1
    for bi,blk in enumerate(BLKS):
        ext=set(q for s in blk for q in NB[s] if q not in blk)
        for u,col in ((BO+bi,0),(BX+bi,1)):
            b[u]=len(ext)-1-32
            for s in blk: W[s*3+col,u]+=8
            for x in ext: W[x*3,u]-=1; W[x*3+1,u]-=1
    Ws['L1']=W; bs['L1']=b
    # L2 seeds
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (src,blk,gate,dst,th,useb) in ((EO,BO,OWN,SO,2,1),(EX,BX,OPP,SX,2,1),(EX,BX,OPP,TX,4,0)):
            u=dst+p; W[src+p,u]+=2
            for r in NB[p]: W[src+r,u]+=1
            W[gate+p,u]+=20; b[u]=-th-20
            if useb:
                for bi in BLK[p]: W[blk+bi,u]+=3
    Ws['L2']=W; bs['L2']=b
    # L3: hop; VX starts = M*(SX hop) + mass hop
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (ch,gate,Bg) in ((SO,OWN,2048),(SX,OPP,2048),(TX,OPP,2048)):
            u=ch+p; W[ch+p,u]+=1
            for r in NB[p]: W[ch+r,u]+=1
            W[gate+p,u]+=Bg; b[u]=-Bg
        u=VX+p; W[SX+p,u]+=M; W[OPP+p,u]+=1
        for r in NB[p]: W[SX+r,u]+=M; W[OPP+r,u]+=1
        W[OPP+p,u]+=16384-1; b[u]=-16384+1
    Ws['L3']=W; bs['L3']=b
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (ch,gate,Bg) in ((SO,OWN,16384),(SX,OPP,16384),(TX,OPP,16384),(VX,OPP,65536)):
            u=ch+p; W[ch+p,u]+=1
            for r in NB[p]: W[ch+r,u]+=1
            W[gate+p,u]+=Bg; b[u]=-Bg
    Ws['L4']=W; bs['L4']=b
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (ch,gate,dst,k) in ((SO,OWN,AO,1),(SX,OPP,AX,1),(TX,OPP,LX,1),(VX,OPP,AXS,13)):
            u=dst+p; W[gate+p,u]+=k; W[ch+p,u]-=1
            for r in NB[p]: W[ch+r,u]-=1
    Ws['L5']=W; bs['L5']=b
    return Ws,bs
