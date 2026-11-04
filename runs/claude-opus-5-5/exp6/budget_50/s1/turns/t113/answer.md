```python
import numpy as np
lg=np.load('/work/pass_26/logits.npy')
print('eye edge (0,1):',lg[0][1],'pass',lg[0][81])
print('eye corner (0,0):',lg[1][0])
print('eye interior (2,2):',lg[2][20])
print('pos4 eyes (0,3),(0,5),(0,7),(0,8):',lg[4][[3,5,7,8]],'pass',lg[4][81], 'moves (7,2)',lg[4][65])
print('pos3 (0,6)',lg[3][6],'(5,4)',lg[3][49])
```