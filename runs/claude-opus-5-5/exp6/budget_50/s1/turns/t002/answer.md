```python
gou = r'''
import numpy as np
N=9
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]
NB=[[ (r+dr)*9+(c+dc) for dr,dc in DIRS if 0<=r+dr<N and 0<=c+dc<N] for r in range(N) for c in range(N)]
# board: np array of 81 ints: 0 empty, 1 black, -1 white
def from_strings(rows):
    b=np.zeros(81,int)
    for r,row in enumerate(rows):
        for c,ch in enumerate(row):
            b[r*9+c]= 1 if ch=='X' else (-1 if ch=='O' else 0)
    return b
def to_strings(b):
    return [''.join('X' if b[r*9+c]==1 else ('O' if b[r*9+c]==-1 else '.') for c in range(9)) for r in range(9)]
def chain(b,p):
    col=b[p]; seen={p}; st=[p]; libs=set()
    while st:
        x=st.pop()
        for y in NB[x]:
            if b[y]==col and y not in seen: seen.add(y); st.append(y)
            elif b[y]==0: libs.add(y)
    return seen,libs
def play(b,p,col):
    """returns new board, captured count; None if illegal (occupied/suicide)"""
    if b[p]!=0: return None,0
    b=b.copy(); b[p]=col; cap=0
    for y in NB[p]:
        if b[y]==-col:
            s,l=chain(b,y)
            if not l:
                for z in s: b[z]=0
                cap+=len(s)
    s,l=chain(b,p)
    if not l: return None,0
    return b,cap
def area_score(b):
    # black area - white area
    sc=b.sum(); seen=set()
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]!=0: border.add(b[y])
            seen|=reg
            if border=={1}: sc+=len(reg)
            elif border=={-1}: sc-=len(reg)
    return sc
'''
open('/work/goutil.py','w').write(gou)
exec(gou)
import json
E='.........'
def blank(): return [list(E) for _ in range(9)]
def put(bd,r,c,ch): bd[r][c]=ch
def fin(bd): return [''.join(x) for x in bd]
pos=[]
# T1 capture single white stone
bd=blank(); put(bd,4,4,'O'); [put(bd,r,c,'X') for r,c in [(3,4),(5,4),(4,3)]]; put(bd,1,1,'O')
pos.append({"board":fin(bd),"to_move":"black"})
# T2 escape
bd=blank(); put(bd,4,4,'X'); [put(bd,r,c,'O') for r,c in [(3,4),(5,4),(4,3)]]; put(bd,7,7,'X')
pos.append({"board":fin(bd),"to_move":"black"})
# T3 L-shape corner atari with shared liberty
bd=blank(); [put(bd,r,c,'O') for r,c in [(0,1),(1,1),(1,0)]]; [put(bd,r,c,'X') for r,c in [(0,2),(1,2),(2,1),(2,0)]]
pos.append({"board":fin(bd),"to_move":"black"})
# T4 long chain 2 libs at ends (false-positive test)
bd=blank(); [put(bd,4,c,'O') for c in range(2,7)]; [put(bd,3,c,'X') for c in range(2,7)]; [put(bd,5,c,'X') for c in range(2,7)]
pos.append({"board":fin(bd),"to_move":"black"})
# T5 eye: black group around (1,1) eye, plus white stones
rows=["XXX......","X.X......","XXX..O...",".........","....O....",".........",".........",".........","........."]
pos.append({"board":rows,"to_move":"black"})
# T6 self-atari: lone point with one liberty for black
rows=[".........",".........",".........","....O....","...O.O...","....O....","......X..",".........","........."]
pos.append({"board":rows,"to_move":"black"})
# T7 empty board black, T8 empty board white
pos.append({"moves":[]}); pos.append({"moves":[40]})
# T9 2-stone white chain in atari in the middle
rows=[".........",".........","...XX....","..XOOX...","...X.....",".........",".........",".........","........."]
pos.append({"board":rows,"to_move":"black"})
# T10 same but white to move (white should escape at (4,4))
pos.append({"board":rows,"to_move":"white"})
for i,p in enumerate(pos):
    if 'board' in p:
        b=from_strings(p['board'])
print(json.dumps({"positions":pos,"activations":True}))
```