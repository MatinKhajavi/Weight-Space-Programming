```python
src=open('/work/builder27.py').read()
old="""    for (r,c) in PTS:
        W[6][LX,L6_AP+P(r,c)]+=-prm['cl']
"""
assert old in src
src=src.replace(old,old+"""    W[6][LX,494]=1.
    Wo[494,81]+=prm.get('passlead',0.)
""")
open('/work/builder31.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v27.json')); prm['passlead']=4.0
json.dump(prm,open('/work/prm_v31.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
print('L6 494 inputs',np.nonzero(W[6][:,494])[0], 'out', Wo[494,81], 'LX bias', B[5][487])
save_all(save,W,B,Wo,bo)
```