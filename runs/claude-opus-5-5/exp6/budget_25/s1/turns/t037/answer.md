```python
code=open('/work/trunk3.py').read()
t4 = r'''
SO=164; TO=245; SX=326; VX=326       # L2: SO,TO,SX ; L3-4: SO,TO,VX
AO=164; LO=245; AX=326; AXS=407      # L5
def build_trunk(M=1024):
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
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (src,blk,gate,dst,th,useb) in ((EO,BO,OWN,SO,2,1),(EO,BO,OWN,TO,4,0),(EX,BX,OPP,SX,2,1)):
            u=dst+p; W[src+p,u]+=2
            for r in NB[p]: W[src+r,u]+=1
            W[gate+p,u]+=20; b[u]=-th-20
            if useb:
                for bi in BLK[p]: W[blk+bi,u]+=3
    Ws['L2']=W; bs['L2']=b
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (ch,gate) in ((SO,OWN),(TO,OWN)):
            u=ch+p; W[ch+p,u]+=1
            for r in NB[p]: W[ch+r,u]+=1
            W[gate+p,u]+=2048; b[u]=-2048
        u=VX+p; W[SX+p,u]+=M
        for r in NB[p]: W[SX+r,u]+=M; W[OPP+r,u]+=1
        W[OPP+p,u]+=2**20; b[u]=-2**20+1
    Ws['L3']=W; bs['L3']=b
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (ch,gate,Bg) in ((SO,OWN,16384),(TO,OWN,16384),(VX,OPP,2**23)):
            u=ch+p; W[ch+p,u]+=1
            for r in NB[p]: W[ch+r,u]+=1
            W[gate+p,u]+=Bg; b[u]=-Bg
    Ws['L4']=W; bs['L4']=b
    W,b=Z(); carry(W,b)
    for p in range(81):
        for (ch,gate,dst,k,sc) in ((SO,OWN,AO,1,1),(TO,OWN,LO,1,1),(VX,OPP,AX,1,1/1024),(VX,OPP,AXS,13,1)):
            u=dst+p; W[gate+p,u]+=k; W[ch+p,u]-=sc
            for r in NB[p]: W[ch+r,u]-=sc
    Ws['L5']=W; bs['L5']=b
    return Ws,bs
'''
code4 = code.split("def build_trunk")[0] + t4
open('/work/trunk4.py','w').write(code4)
g={}; exec(code4,g); Ws,bs=g['build_trunk']()
import numpy as np
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
vals=set()
for k in Ws:
    w=Ws[k][Ws[k]!=0]
    # check bf16 exactness
    import struct
    bad=[x for x in np.unique(w) if (np.frombuffer(np.float32(x).tobytes(),np.uint32)[0] & 0xFFFF)!=0]
    print(k,'non-bf16 weights:',bad[:5])
```