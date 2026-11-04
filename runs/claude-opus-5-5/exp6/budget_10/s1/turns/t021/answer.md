```python
exec(open('/work/build3.py').read())
layers,OW,Ob=build(DEFAULT)
save_all(layers,OW,Ob,save)
print('saved v3', saved())
```