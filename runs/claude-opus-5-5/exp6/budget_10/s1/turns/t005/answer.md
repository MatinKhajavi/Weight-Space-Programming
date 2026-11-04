```python
import numpy as np
d='/work/pass_1/'
lg=np.load(d+'logits.npy'); pr=np.load(d+'probs.npy'); lgl=np.load(d+'legal.npy')
np.set_printoptions(linewidth=200, precision=1, suppress=True)
for i in [0,2,4,6,8]:
    print('pos',i,'pass logit',lg[i,81],'pass prob',pr[i,81])
    print(lg[i,:81].reshape(9,9))
```