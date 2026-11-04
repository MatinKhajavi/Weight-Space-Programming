
import json, numpy as np
def load_pass(k):
    d='/work/pass_%d/'%k
    G=json.load(open(d+'games.json')); boards=np.load(d+'boards.npy')
    return G, boards
def pb(b):
    s=''
    for r in range(9):
        s+=''.join('X' if b[r*9+c]==1 else 'O' if b[r*9+c]==-1 else '.' for c in range(9))+'\n'
    return s
