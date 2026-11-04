```python
exec(open('/work/ana.py').read())
from collections import Counter
import os
for k in (2,3,5,6,8,9,10,13,14,15):
    G,B,V=load_pass(k)
    s=[g for g in G if g['opponent']=='strong']
    np_=[g for g in s if g['end']!='repetition']
    rep=[g for g in s if g['end']=='repetition']
    w=sum(g['result']>0 for g in np_)
    print(k,'n',len(s),'winrate %.3f'%np.mean([g['result']>0 for g in s]),'nonrep wr %.3f (%d)'%(w/len(np_),len(np_)),
          'rep W/L',sum(g['result']>0 for g in rep),sum(g['result']<0 for g in rep),
          'B %.2f W %.2f'%(np.mean([g['result']>0 for g in s if g['net_color']=='black']),np.mean([g['result']>0 for g in s if g['net_color']=='white'])))
```