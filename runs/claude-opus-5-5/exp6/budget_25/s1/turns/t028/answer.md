```python
import json
exec(open('/work/ana.py').read())
G,B,V=load_pass(3)
req=json.load(open('/work/test1_req.json')); poss=req['positions'][:30]
extra=[]
def bstr(b):
    return [''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in b[r*9:r*9+9]) for r in range(9)]
for gi,ply in ((32,73),(39,65),(44,81),(53,90),(17,105),(47,105)):
    g=G[gi]; mv=g['moves'][ply+1]
    assert mv['by']=='net',(gi,ply)
    col='black' if g['net_color']=='black' else 'white'
    extra.append({"board":bstr(B[gi,ply]),"to_move":col})
allp=poss+extra
json.dump({"positions":allp},open('/work/test2_req.json','w'))
print(json.dumps({"positions":allp,"activations":True},separators=(',',':')))
```