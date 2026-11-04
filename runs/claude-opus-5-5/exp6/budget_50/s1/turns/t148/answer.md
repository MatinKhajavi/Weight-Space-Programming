```python
import json, numpy as np
from collections import defaultdict
st=defaultdict(lambda:[0,0])
for dd in ('/work/pass_33/','/work/pass_34/','/work/pass_35/','/work/pass_36/','/work/pass_37/','/work/pass_31/','/work/pass_32/','/work/pass_27/'):
    G=json.load(open(dd+'games.json'))
    for g in G:
        if g['opponent']!='strong' or g['net_color']!='black': continue
        m0=g['moves'][0]['move']
        st[divmod(m0,9)][0]+=g['result']==1; st[divmod(m0,9)][1]+=1
for k,v in sorted(st.items(), key=lambda x:-x[1][1])[:12]: print(k, v, round(v[0]/v[1],2))
```