```python
gocode = r'''
import numpy as np, random
def nb(p):
    r,c=divmod(p,9); out=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(rr*9+cc)
    return out
NB=[nb(p) for p in range(81)]
def group(bd,p):
    col=bd[p]; st=[p]; seen={p}; libs=set()
    while st:
        q=st.pop()
        for r in NB[q]:
            if bd[r]==0: libs.add(r)
            elif bd[r]==col and r not in seen: seen.add(r); st.append(r)
    return seen,libs
def play(bd,p,col):
    # bd: list of 81 ints 0 empty,1 black,2 white; returns new board or None if illegal(suicide/occupied)
    if bd[p]!=0: return None
    b=list(bd); b[p]=col; opp=3-col; cap=0
    for r in NB[p]:
        if b[r]==opp:
            g,l=group(b,r)
            if not l:
                for q in g: b[q]=0
                cap+=len(g)
    g,l=group(b,p)
    if not l: return None
    return b
def libcounts(bd):
    lc=[0]*81
    for p in range(81):
        if bd[p]: lc[p]=len(group(bd,p)[1])
    return lc
def tostr(bd):
    return [''.join('.XO'[bd[r*9+c]] for c in range(9)) for r in range(9)]
def random_game(n,seed):
    rng=random.Random(seed); bd=[0]*81; col=1; hist=[]
    for i in range(n):
        moves=[p for p in range(81) if bd[p]==0]
        rng.shuffle(moves); done=False
        for p in moves:
            nb2=play(bd,p,col)
            if nb2 is None: continue
            # avoid filling own eye
            if all(bd[r]==col for r in NB[p]) : continue
            bd=nb2; hist.append(p); done=True; break
        if not done: hist.append('pass')
        col=3-col
    return bd,hist,col
'''
open('/work/go.py','w').write(gocode)
exec(gocode)
import json
poss=[]; meta=[]
for s in range(60):
    n=random.Random(1000+s).randint(10,70)
    bd,hist,col=random_game(n,s)
    poss.append({"board":tostr(bd),"to_move":"black" if col==1 else "white"})
    meta.append((bd,col))
json.dump({'meta':meta},open('/work/test1_meta.json','w'))
req={"positions":poss,"activations":True}
open('/work/test1_req.json','w').write(json.dumps(req))
print(len(json.dumps(req)))
print(json.dumps(poss[0]))
```