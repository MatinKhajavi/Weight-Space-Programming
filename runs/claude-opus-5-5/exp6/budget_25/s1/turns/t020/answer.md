```python
src=open('/work/build.py').read()
# v2: add 2x2 block seeds at L1 (units 326..389 own, 390..453 opp), used in L2 s2 seeds; bigger gates
src=src.replace("""    Ws['L1']=W; bs['L1']=b
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32); carry(W,b)""","""    for bi in range(64):
        r0,c0=divmod(bi,8); blk=[(r0+i)*9+c0+j for i in (0,1) for j in (0,1)]
        ext=set(q for s in blk for q in NB[s] if q not in blk)
        for u,col in ((326+bi,0),(390+bi,1)):
            b[u]=len(ext)-1-32
            for s in blk: W[s*3+col,u]+=8
            for x in ext: W[x*3+0,u]-=1; W[x*3+1,u]-=1
    Ws['L1']=W; bs['L1']=b
    BLK=[[bi for bi in range(64) if p in [(bi//8+i)*9+bi%8+j for i in (0,1) for j in (0,1)]] for p in range(81)]
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32); carry(W,b)""")
src=src.replace("""            W[gate+p,u]+=20; b[u]=-th-20
    Ws['L2']=W""","""            W[gate+p,u]+=20; b[u]=-th-20
            if th==2:
                for bi in BLK[p]: W[(326 if src==A else 390)+bi,u]+=3
    Ws['L2']=W""")
src=src.replace("for L,Bg in (('L3',500),('L4',4000)):","for L,Bg in (('L3',2048),('L4',16384)):")
open('/work/build.py','w').write(src)
g=dict(save=save); exec(src,g)
Ws,bs=g['build_all'](g['P0'])
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
g['save_all'](g['P0'])
open('/work/notes.txt','a').write("v2: added 2x2 block seeds (L1 326-453), gates 2048/16384. pass2 baseline v1: random 100%, weak 97%, strong 52% (11 repetition losses due to false atari->self-atari)\n")
```