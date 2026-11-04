```python
import numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_1/'
lg=np.load(d+'logits.npy'); pr=np.load(d+'probs.npy')
print(lg[4][[10,11,19,20,22]], lg[4][81])  # eye (1,1)=10
print(lg[5][[40,39,41]], lg[5][81])  # self-atari (4,4)=40
a6=np.load(d+'act_L6.npy')
print('SA at 40:',a6[5][324+40],'EYE at 10:',a6[4][405+10])
print(np.round(lg[6][:81].reshape(9,9),1))
```