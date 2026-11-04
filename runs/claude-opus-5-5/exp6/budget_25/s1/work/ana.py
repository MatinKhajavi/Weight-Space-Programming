
import json, numpy as np
def load_pass(k):
    G=json.load(open(f'/work/pass_{k}/games.json'))
    B=np.load(f'/work/pass_{k}/boards.npy'); V=np.load(f'/work/pass_{k}/valid.npy')
    return G,B,V
def sb(b): return '\n'.join(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in b[r*9:r*9+9]) for r in range(9))
def sb2(b1,b2):
    a=sb(b1).split('\n'); c=sb(b2).split('\n')
    return '\n'.join(x+'   '+y for x,y in zip(a,c))
