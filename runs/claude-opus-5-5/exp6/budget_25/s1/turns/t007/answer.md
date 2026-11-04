```python
req={"positions":poss[:30]+[{"board":["........."]*9,"to_move":"black"}],"activations":True}
print(json.dumps(req,separators=(',',':')))
```