```python
src=open('/work/builder19.py').read()
src=src.replace("""    W[1][I(0,0,2),486]=1.
    for k in (2,3,4,5): W[k][486,486]=1.
""","""    W[1][I(0,0,2),486]=1.
    for k in (2,3,4,5): W[k][486,486]=1.
    # LEADX at L5 unit 487 from L4 carry
    LX=487
    for (r,c) in PTS:
        p=P(r,c); W[5][OWN+p,LX]+=1.; W[5][OPP+p,LX]+=-1.
    W[5][486,LX]+=15.; B[5][LX]=-7.5-prm['TL']
    for (r,c) in PTS:
        W[6][LX,L6_AP+P(r,c)]+=-prm['cl']
""")
open('/work/builder21.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v20.json')); prm.update(TL=15.,cl=1/10.)
json.dump(prm,open('/work/prm_v21.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
print(prm)
```