```python
import numpy as np, json
exec(open('/work/builder5.py').read())
d='/work/pass_13/'
lg=np.load(d+'logits.npy'); A={k:np.load(d+'act_L%d.npy'%k) for k in range(1,7)}
pos=json.load(open('/work/testpos3.json'))
for i in (0,1):
    print('pos',i,'logit (2,0)',lg[i][18],'pass',lg[i][81])
    print('L6 AP'); print(np.round(A[6][i][L6_AP:L6_AP+81].reshape(9,9),2))
    print('L5 XO'); print(np.round(A[5][i][XO:XO+81].reshape(9,9),3))
    print('L5 XE'); print(np.round(A[5][i][XE:XE+81].reshape(9,9),3))
    print('SA',A[6][i][L6_SA+18],'LIN',A[6][i][L6_LIN+18]-10,'OT',A[6][i][L6_OT+18])
```