```python
exec(open('/work/build.py').read())
P = dict(DEFAULT)
P.update(Wc=14.0, Wc2=3.0, We=10.0, We2=2.0, Wa=3.0, Wd=2.0, Wsa=25.0)
layers, OW, Ob = build(P)
save_all(layers, OW, Ob, save)
import json
json.dump({k:(v if k!='kern' else {str(kk):vv for kk,vv in v.items()}) for k,v in P.items()}, open('/work/P_v2.json','w'))
print('saved v2')
```