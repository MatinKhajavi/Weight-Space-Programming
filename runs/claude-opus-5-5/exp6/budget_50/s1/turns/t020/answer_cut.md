```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
gi=64; g=G[gi]; print(g['net_color'], g['margin'])
for k in range(0,60,2):
    m=g['moves'][k+1] if g['net_color']=='white' else g['moves'][k]
for k in (11,21,31,41,51,