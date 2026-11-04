```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(5)
wl=[g for g in G if g['opponent']=='weak' and g['result']<0]
for g in wl:
    gi=g['game']; n=len(g['moves'])
    print(gi,g['net_color'],g['margin'],g['end'],'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'],'len',n)
    print(sb(B[gi,n-1]));print()
```