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
    for gi in range(len(G)):
        for ply in range(0,162,3):
            if not valid[gi,ply]: continue
            b=boards[gi,ply].astype(int)
            ch,idx=chains(b)
            for col,mem,libs in ch:
                if len(libs)<2: continue
                lib_of={s:[y for y in NBR[s] if b[y]==0] for s in mem}
                for q in mem:
                    if not lib_of[q]: continue
                    dist=chain_dist(mem,q)
                    # evidence classes within reach R
                    for R in (4,6,8):
                        cls=set(); e1=False
                        for s,dd in dist.items():
                            if dd<=R:
                                if lib_of[s]: cls.add(PARv[s])
                                if len(lib_of[s])>=2: e1=True
                        ok = e1 or len(cls)==2
                        st[(R,ok)]+=1
                        if not ok: st[(R,'big' if len(mem)>5 else 'small')]+=1
print(st)
```