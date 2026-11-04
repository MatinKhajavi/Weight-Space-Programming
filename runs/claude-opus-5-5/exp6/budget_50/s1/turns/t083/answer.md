```python
import json, numpy as np
exec(open('/work/builder12.py').read())
prm=json.load(open('/work/prm_v12.json'))
W,B,Wo,bo=build(prm)
print(prm)
p=P(4,4)
nz=np.nonzero(Wo[:,p])[0]
print([(int(i),float(Wo[i,p])) for i in nz])
print('bias',bo[:9],bo[81])
# check L6 units feeding: AP block weights from L5
print('AP unit 40 weights nonzero', [(int(i),float(W[6][i,L6_AP+40])) for i in np.nonzero(W[6][:,L6_AP+40])[0]], B[6][L6_AP+40])
print('cycle unit 324', [(int(i),float(W[6][i,324])) for i in np.nonzero(W[6][:,324])[0]], B[6][324], np.nonzero(Wo[324])[0])
```