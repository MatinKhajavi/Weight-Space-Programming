```python
src=open('/work/build6.py').read()
src=src.replace("prox_opp={(0,1):0.5,(1,1):0.3}, prox_own={})","prox_opp={(0,1):0.5,(1,1):0.3}, prox_own={(1,1):0.4,(0,2):0.3}, center_ph=[0.0,0.0,0.0,0.02,0.04])")
src=src.replace("""        OW[E6,p] -= P['edgeph'][LINE[p]]""","""        OW[E6,p] -= P['edgeph'][LINE[p]]
        OW[E6,p] += P['center_ph'][LINE[p]]""")
src=src.replace("Wt=1.0, w=2.0, k0=1.0, P0=0.5,","Wt=1.0, w=2.0, k0=1.0, P0=1.0,")
open('/work/build7.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers], DEFAULT['P0'], DEFAULT['prox_own'])
save_all(layers,OW,Ob,save)
print('saved v7')
```