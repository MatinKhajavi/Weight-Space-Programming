```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
dirs={}
for h in range(6):
    q,k,v,o=headmats(0,h)
    for name,M in [('q',q),('k',k),('v',v)]:
        u,s,vt=np.linalg.svd(M)
        dirs[f'{h}{name}0']=vt[0]; dirs[f'{h}{name}1']=vt[1]
names=list(dirs)
D=np.array([dirs[n] for n in names])
C=np.abs(D@D.T)
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
print(pd.DataFrame(C,index=names,columns=names).round(2).loc[[n for n in names if n[0] in '1245'],[n for n in names if n[0] in '1245']])
```