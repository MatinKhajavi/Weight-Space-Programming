```python
import numpy as np
W6=load('L6.W'); b6=load('L6.b'); Wo=load('out.W'); bo=load('out.b')
p=9
for slot,name in ((0,'LIN'),(81,'ATT'),(162,'SELF'),(243,'EXL'),(324,'TERR'),(405,'KO')):
    u=slot+p
    nz=np.nonzero(W6[:,u])[0]
    print(name,'out w',Wo[u,p],'bias',b6[u],'n_in',len(nz), [(int(i),float(W6[i,u])) for i in nz][:12])
print('pass bias',bo[81], 'bo[p]',bo[p])
```