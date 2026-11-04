
import numpy as np
NBR=[]
for p in range(81):
    r,c=divmod(p,9); l=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: l.append(rr*9+cc)
    NBR.append(l)
def chains(b):
    seen=-np.ones(81,int); out=[]
    for p in range(81):
        if b[p]!=0 and seen[p]<0:
            col=b[p]; st=[p]; seen[p]=len(out); mem=[]; libs=set()
            while st:
                x=st.pop(); mem.append(x)
                for y in NBR[x]:
                    if b[y]==0: libs.add(y)
                    elif b[y]==col and seen[y]<0: seen[y]=len(out); st.append(y)
            out.append((col,mem,libs))
    return out, seen
def play(b,p,col):
    # returns new board, captured count, or None if illegal(suicide/occupied)
    if b[p]!=0: return None,0
    nb=b.copy(); nb[p]=col; cap=0
    ch,idx=chains(nb)
    for y in NBR[p]:
        if nb[y]==-col:
            c_,mem,libs=ch[idx[y]]
            if len(libs)==0:
                for m in mem:
                    if nb[m]!=0: nb[m]=0; cap+=1
    ch,idx=chains(nb)
    if len(ch[idx[p]][2])==0: return None,0
    return nb,cap
def libs_after(b,p,col):
    nb,cap=play(b,p,col)
    if nb is None: return None,0
    ch,idx=chains(nb)
    return len(ch[idx[p]][2]),cap
