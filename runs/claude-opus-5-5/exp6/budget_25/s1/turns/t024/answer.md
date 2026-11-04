```python
import numpy as np
def walks(stones, k=3):
    idx={s:i for i,s in enumerate(stones)}; n=len(stones)
    A=np.eye(n)
    for s in stones:
        r,c=s
        for d in ((0,1),(1,0),(0,-1),(-1,0)):
            t=(r+d[0],c+d[1])
            if t in idx: A[idx[s],idx[t]]=1
    v=np.linalg.matrix_power(A,k)@np.ones(n)
    return v.astype(int)
shapes={'1':[(0,0)],'2':[(0,0),(0,1)],'I3':[(0,i) for i in range(3)],'L3':[(0,0),(0,1),(1,1)],
'I4':[(0,i) for i in range(4)],'sq':[(0,0),(0,1),(1,0),(1,1)],'T4':[(0,0),(0,1),(0,2),(1,1)],
'I5':[(0,i) for i in range(5)],'I6':[(0,i) for i in range(6)],'I8':[(0,i) for i in range(8)],
'2x3':[(i,j) for i in range(2) for j in range(3)],'2x4':[(i,j) for i in range(2) for j in range(4)],
'3x3':[(i,j) for i in range(3) for j in range(3)],'L5':[(0,0),(0,1),(0,2),(1,2),(2,2)]}
for k,s in shapes.items(): print(k, walks(s).tolist())
```