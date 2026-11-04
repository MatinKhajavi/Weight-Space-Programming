```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
print(len(G), G[0].keys())
for g in G:
    if g['end']!='two_passes' or g['result']<0:
        print(g['opponent'], g['net_color'], g['result'], g['margin'], g['end'], len(g['moves']), g.get('captured'), g.get('lost'))
```