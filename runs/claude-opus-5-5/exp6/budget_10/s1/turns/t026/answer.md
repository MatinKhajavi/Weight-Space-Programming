```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
PARv=[(p//9+p%9)%2 for p in range(81)]
def evid(b,col,mem,libs):
    memset=set(mem)
    lib_of={s:set(y for y in NBR[s] if b[y]==0) for s in mem}
    ev={}
    pars=set(PARv[s] for s in mem if lib_of[s])
    ev['par']=len(pars)==2
    ev['e1']=any(len(lib_of[s])>=2 for s in mem)
    T=False; star=False; tri=False
    for s in mem:
        nb=[t for t in NBR[s] if t in memset]
        if len(lib_of[s])+sum(len(lib_of[t]) for t in nb)>=2: T=True
        if len(lib_of[s].union(*[lib_of[t] for t in nb]))>=2: star=True
        r,c=divmod(s,9)
        for d in (1,9):
            a,bq=s-d,s+d
            if d==1 and (c==0 or c==8): continue
            if d==9 and (r==0 or r==8): continue
            if a in memset and bq in memset and len(lib_of[a]|lib_of[s]|lib_of[bq])>=2: tri=True
    ev['T']=T; ev['star']=star; ev['tri']=tri
    return ev
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
                ev=evid(b,col,mem,libs)
                key='atari' if len(libs)==1 else 'safe'
                st[key]+=1
                sz='small' if len(mem)<=3 else 'big'
                st[(key,sz)]+=1
                for combo,name in ((ev['par'] or ev['e1'],'par+e1'),(ev['par'] or ev['e1'] or ev['tri'],'par+e1+tri'),(ev['par'] or ev['e1'] or ev['star'],'par+e1+star'),(ev['T'],'T'),(ev['par'] or ev['e1'] or ev['T'],'par+e1+T')):
                    st[(key,sz,name,combo)]+=1
for kk in sorted(st, key=str): print(kk, st[kk])
```