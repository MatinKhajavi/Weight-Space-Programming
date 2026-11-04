```python
import numpy as np
exec(open('/work/builder3.py').read())
d='/work/pass_4/'
lg=np.load(d+'logits.npy')
A={k:np.load(d+'act_L%d.npy'%k) for k in range(1,7)}
def show(arr,name):
    print(name); print(np.round(arr.reshape(9,9),2))
i=0
print('logits pos0', lg[0][[P(4,5),P(1,2),P(2,1)]])
show(A[1][i][EP:EP+81],'L1 EP'); show(A[1][i][OP:OP+81],'L1 OP')
show(A[6][i][L6_AP:L6_AP+81],'L6 AP')
i=4
show(A[1][i][EP:EP+81],'p4 L1 EP'); show(A[1][i][OP:OP+81],'p4 L1 OP')
show(A[5][i][EP:EP+81],'p4 L5 EP'); show(A[5][i][OP:OP+81],'p4 L5 OP')
show(A[6][i][L6_AP:L6_AP+81],'p4 L6 AP')
```