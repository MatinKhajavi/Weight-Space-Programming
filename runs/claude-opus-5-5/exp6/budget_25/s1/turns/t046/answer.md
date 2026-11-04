```python
t5=open('/work/trunk5.py').read()
base=open('/work/trunk3.py').read().split("OWN=0;")[0]
extra = r'''
BLKS=[[(r0+i)*9+c0+j for i in (0,1) for j in (0,1)] for r0 in range(8) for c0 in range(8)]
BLK=[[bi for bi in range(64) if p in BLKS[bi]] for p in range(81)]
def Z(n=512): return np.zeros((n,512),np.float32), np.zeros(512,np.float32)
'''
t5=t5.replace(base, base+extra,1)
open('/work/trunk5.py','w').write(t5)
g={}; exec(t5,g); Ws,bs=g['build_trunk']()
import numpy as np
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
for k in Ws:
    w=np.unique(Ws[k][Ws[k]!=0]); bad=[x for x in w if (np.frombuffer(np.float32(x).tobytes(),np.uint32)[0] & 0xFFFF)!=0]
    print(k,len(bad))
```