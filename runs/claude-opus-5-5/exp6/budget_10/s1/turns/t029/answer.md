```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
PARv=[(p//9+p%9)%2 for p in range(81)]
def chain_dist(mem,src):
    memset=set(mem); dist={src:0}; fr=[src]
    while fr:
        nf=[]
        for x in fr:
            for y in NBR[x]:
                if y in memset and y not in dist: dist[y]=dist[x]+1; nf.append(y)
        fr=nf
    return dist
st=collections.Counter()
for k in (2,3):
    G,boards=load_pass(k)
    valid=np.load('/work/pass_%d/valid.npy'%k)
    for gi in range(0,len(G),2):
        for ply in range(0,162,4):
            if not valid[gi,ply]: continue
            b=boards[gi,ply].astype(int)
            ch,idx=chains(b)
            for col,mem,libs in ch:
                if len(libs)<2: continue
                memset=set(mem)
                lib_of={s:set(y for y in NBR[s] if b[y]==0) for s in mem}
                # local evidences located at stones
                tri={}; star={}
                for s in mem:
                    nb=[t for t in NBR[s] if t in memset]
                    star[s]= len(lib_of[s].union(*[lib_of[t] for t in nb]))>=2
                    r,c=divmod(s,9); tr=False
                    for d in (1,9):
                        if (d==1 and (c==0 or c==8)) or (d==9 and (r==0 or r==8)): continue
                        a,bq=s-d,s+d
                        if a in memset and bq in memset and len(lib_of[a]|lib_of[s]|lib_of[bq])>=2: tr=True
                    tri[s]=tr
                for q in mem:
                    if not lib_of[q]: continue
                    dist=chain_dist(mem,q)
                    for R in (3,4):
                        cls=set(); e1=False; t_=False; s_=False
                        for s,dd in dist.items():
                            if dd<=R:
                                if lib_of[s]: cls.add(PARv[s])
                                if len(lib_of[s])>=2: e1=True
                            if dd<=R-1:   # tri/star computed one layer later -> reach one less
                                if tri[s]: t_=True
                                if star[s]: s_=True
                        base= e1 or len(cls)==2
                        st[(R,'par+e1',base)]+=1
                        st[(R,'par+e1+tri(R-1)',base or t_)]+=1
                        st[(R,'par+e1+star(R-1)',base or s_)]+=1
for kk in sorted(st,key=str): print(kk,st[kk])
```