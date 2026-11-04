```python
import numpy as np
B=np.load('/work/pass_2/boards.npy'); V=np.load('/work/pass_2/valid.npy')
def show(b):
    return '\n'.join(''.join('.XO'[int(v)] if v>=0 else 'O' for v in b[r*9:(r+1)*9]) for r in range(9))
def sb(b): return '\n'.join(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in b[r*9:r*9+9]) for r in range(9))
reps=[g for g in G if g['end']=='repetition' and g['result']<0]
for g in reps[:4]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    print('game',gi,g['net_color'],'margin',g['margin'],'len',n)
    print([(m['by'][0],m['move'],m.get('top5',[[None,0]])[0]) for m in mv[-10:]])
    print(sb(B[gi,n-1]))
```