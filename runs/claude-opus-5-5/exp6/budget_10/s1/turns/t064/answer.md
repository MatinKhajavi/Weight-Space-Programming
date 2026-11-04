```python
src=open('/work/build7.py').read()
src=src.replace("""    Ob[81]=P['P0']
    return layers,OW,Ob""","""    Ob[81]=P['P0']
    OW*=P.get('scale',1.0); Ob*=P.get('scale',1.0)
    return layers,OW,Ob""")
src=src.replace("Wt=1.0, w=2.0, k0=1.0, P0=1.0,","Wt=1.0, w=2.0, k0=1.0, P0=1.0, scale=2.0,")
open('/work/build8.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers], DEFAULT['scale'], Ob[81])
save_all(layers,OW,Ob,save)
print('saved v8')
```