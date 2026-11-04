
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
