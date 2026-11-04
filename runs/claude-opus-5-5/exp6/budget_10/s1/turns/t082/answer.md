```python
src=open('/work/build9.py').read()
src=src.replace("P0=1.0, scale=3.5,","P0=1.0, scale=5.0,")
open('/work/build10.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print(DEFAULT['scale'])
save_all(layers,OW,Ob,save)
print('saved v10')
```