```python
ana = r'''
import json, numpy as np
def load_pass(k):
    G=json.load(open(f'/work/pass_{k}/games.json'))
    B=np.load(f'/work/pass_{k}/boards.npy'); V=np.load(f'/work/pass_{k}/valid.npy')
    return G,B,V
def sb(b): return '\n'.join(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in b[r*9:r*9+9]) for r in range(9))
def sb2(b1,b2):
    a=sb(b1).split('\n'); c=sb(b2).split('\n')
    return '\n'.join(x+'   '+y for x,y in zip(a,c))
'''
open('/work/ana.py','w').write(ana)
exec(ana)
G,B,V=load_pass(2)
reps=[g for g in G if g['end']=='repetition' and g['result']<0]
for g in reps[:4]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    print('game',gi,g['net_color'],'margin',g['margin'],'len',n)
    print([(m['by'][0],m['move'],m.get('top5',[[None,0]])[0]) for m in mv[-10:]])
    print(sb2(B[gi,n-3],B[gi,n-1]))
```