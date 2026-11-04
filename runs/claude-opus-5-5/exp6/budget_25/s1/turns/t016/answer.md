```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(2)
ls=[g for g in G if g['opponent']=='strong' and g['result']<0 and g['end']=='two passes']
for g in ls[:6]:
    gi=g['game']; n=g['moves']
    print(gi,g['net_color'],g['margin'],'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'],'len',n)
    print(sb(B[gi,n-1])); print()
```