```python
go = r'''
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
'''
open('/work/go.py','w').write(go)
exec(go)
exec(open('/work/an.py').read())
G,boards=load_pass(2)
import collections
stats=collections.Counter()
for gi,g in enumerate(G):
    mv=g['moves']; netcol = 1 if g['net_color']=='black' else -1
    for k,m in enumerate(mv):
        if m['by']!='net' or m['move']==81: continue
        b = boards[gi,k-1] if k>0 else np.zeros(81,int)
        b=b.astype(int)
        p=m['move']
        la,cap=libs_after(b,p,netcol)
        # available captures
        ch,idx=chains(b)
        caps=[ (len(mem),list(libs)[0]) for col,mem,libs in ch if col==-netcol and len(libs)==1]
        best=max([c[0] for c in caps],default=0)
        stats['moves']+=1
        if best>0:
            stats['capture_available']+=1
            if cap>0: stats['captured_when_avail']+=1
        if la is not None and la==1 and cap==0: stats['selfatari_nocap']+=1
        if la is not None and la==1 and cap>0: stats['selfatari_cap']+=1
        # own in atari
        own_at=[ (len(mem),list(libs)[0]) for col,mem,libs in ch if col==netcol and len(libs)==1]
        if own_at: stats['own_atari']+=1
print(stats)
```