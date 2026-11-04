```python
exec(open('/work/ana.py').read())
from collections import Counter
G,B,V=load_pass(20)
for g in [g for g in G if g['end']=='repetition']:
    gi=g['game']; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
    k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
    seq=[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if 'top5' in m else '') for m in mv[k0+1:n]]
    print(gi,g['result'],g['margin'],n-1-k0,seq)
ls=[g for g in G if g['result']<0]
print(sorted(g['margin'] for g in ls))
# how many net passes occurred while a capture (true) was available
exec(open('/work/go.py').read())
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
cnt=Counter()
for g in G:
    gi=g['game']; me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']=='net' and m['move']==81:
            bd=tob(B[gi,k-1]); 
            caps=[]
            for p in range(81):
                if bd[p]: continue
                nb2=play(bd,p,me)
                if nb2 is None: continue
                c=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
                if c: caps.append(c)
            cnt['pass_with_cap' if caps else 'pass_nocap']+=1
            if caps and max(caps)>=2: cnt['pass_with_bigcap']+=1
print(cnt)
```