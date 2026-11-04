```python
import json
allp=json.load(open('/work/test2_req.json'))['positions']
print(json.dumps({"positions":allp,"activations":True},separators=(',',':'))[:100])
```