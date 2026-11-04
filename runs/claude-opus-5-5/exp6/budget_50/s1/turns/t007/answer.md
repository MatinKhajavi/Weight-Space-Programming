```python
reps=[g for g in G if g['end']=='repetition' and g['result']==-1]
B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
for g in reps[:4]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    print('game',gi,g['opponent'],g['net_color'],'len',n)
    print([(m['by'][0],m['move']) for m in mv[-12:]])
    # find which earlier board equals last
    last=B[gi,n-1]
    for k in range(n-2,max(-1,n-12),-1):
        if (B[gi,k]==last).all(): print(' repeats board after ply',k)
    for s in to_strings(B[gi,n-1]): print('   ',s)
    print('net top5 at last move',mv[-1].get('top5'))
```