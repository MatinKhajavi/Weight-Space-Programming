```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(5)
def area(b):
    # returns black area, white area, neutral empties
    seen=set(); ba=int((b==1).sum()); wa=int((b==-1).sum()); neu=0
    for p in range(81):
        if b[p]==0 and p not in seen:
            st=[p]; seen.add(p); reg=[]; cols=set()
            while st:
                x=st.pop(); reg.append(x)
                for y in NBR[x]:
                    if b[y]==0 and y not in seen: seen.add(y); st.append(y)
                    elif b[y]!=0: cols.add(b[y])
            if cols=={1}: ba+=len(reg)
            elif cols=={-1}: wa+=len(reg)
            else: neu+=len(reg)
    return ba,wa,neu
rows=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    n=len(g['moves']); b=boards[gi,n-1].astype(int)
    ba,wa,neu=area(b)
    # who passed first in final double pass
    rows.append((g['result'],round(g['margin']),neu))
rows.sort(key=lambda r:r[1])
print(rows)
```