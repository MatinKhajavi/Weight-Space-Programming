```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
G,B,V=load_pass(20)
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
for g in G:
    if g['result']<0 and g['margin']==-0.5:
        gi=g['game']; mv=g['moves']; n=len(mv)
        lastp=[k for k,m in enumerate(mv) if m['by']=='net' and m['move']==81][-1]
        print(gi,g['net_color'],'last net pass ply',lastp,'of',n)
        print(sb(B[gi,lastp-1]))
        print([(divmod(a,9) if a<81 else 'P',p) for a,p in mv[lastp]['top5']])
        bd=tob(B[gi,lastp-1]); me=1 if g['net_color']=='black' else 2
        for p in range(81):
            if bd[p]: continue
            cols={bd[q] for q in NB[p]}-{0}
            if cols=={1,2}:
                nb2=play(bd,p,me)
                print('  dame',divmod(p,9),'legal' if nb2 else 'illegal', 'libs after',len(group(nb2,p)[1]) if nb2 else None)
```