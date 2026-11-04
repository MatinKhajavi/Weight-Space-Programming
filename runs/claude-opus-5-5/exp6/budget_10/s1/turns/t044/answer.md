```python
exec(open('/work/build5.py').read())
layers,OW,Ob=build(DEFAULT)
save_all(layers,OW,Ob,save)
print('saved v5')
```