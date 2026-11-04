```python
import json
exec(open('/work/ana.py').read())
G,B,V=load_pass(10)
def bstr(b): return [''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in b[r*9:r*9+9]) for r in range(9)]
reps=[g for g in G if g['end']=='repetition' and g['result']<0]
pos=[]; meta=[]
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
    k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
    pos.append({"board":bstr(B[gi,k0]),"to_move":g['net_color']}); meta.append((gi,k0+1,mv[k0+1]['move']))
json.dump({'pos':pos,'meta':meta},open('/work/test3_req.json','w'))
print(len(pos)); print(json.dumps({"positions":pos,"activations":True},separators=(',',':')))
```