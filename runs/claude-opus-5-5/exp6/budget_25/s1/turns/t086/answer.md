```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(17)
ls=[g for g in G if g['result']<0]
print(sorted(g['margin'] for g in ls))
print('lost stones in losses',np.mean([g['stones_lost_by_net'] for g in ls]),'captured',np.mean([g['stones_captured_by_net'] for g in ls]))
ws=[g for g in G if g['result']>0]
print('lost stones in wins',np.mean([g['stones_lost_by_net'] for g in ws]),'captured',np.mean([g['stones_captured_by_net'] for g in ws]))
for g in ls[:4]:
    gi=g['game']; n=len(g['moves'])
    print(gi,g['net_color'],g['margin'],g['end'],'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
    print(sb(B[gi,n-1]))
```