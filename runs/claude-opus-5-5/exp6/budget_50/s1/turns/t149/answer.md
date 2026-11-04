```python
src=open('/work/builder24.py').read()
old="""        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']"""
assert old in src
src=src.replace(old,"""        for dr,dc in ((2,0),(-2,0),(0,2),(0,-2)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: W[6][OWN+P(rr,cc),L6_LIN+p]+=prm.get('jump',0.)
        for dr,dc in ((1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: W[6][OWN+P(rr,cc),L6_LIN+p]+=prm.get('knight',0.)
        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']""")
open('/work/builder27.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v24.json')); prm.update(jump=0.8,knight=0.4)
json.dump(prm,open('/work/prm_v27.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```