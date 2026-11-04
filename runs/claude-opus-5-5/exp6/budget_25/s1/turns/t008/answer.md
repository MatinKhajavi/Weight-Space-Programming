```python
import json
req=json.load(open('/work/test1_req.json'))
poss=req['positions']
req={"positions":poss[:30]+[{"board":["........."]*9,"to_move":"black"}],"activations":True}
print(json.dumps(req,separators=(',',':')))
```