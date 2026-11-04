```python
import json
P1=json.load(open('/work/testpos.json'))
P1.append({"board":[".........",".........","....O....","...OXXO..","....O....",".........",".........",".........","........."],"to_move":"black"})  # black 2-stone chain libs (4,5),(2,5)?
P1.append({"board":[".........",".........","....X....","...XOOX..","....X....",".........",".........",".........","........."],"to_move":"black"})  # white 2-lib chain
P1.append({"board":[".........",".XXX.....",".X.X.....",".XXX.....",".........",".........",".........",".........","........."],"to_move":"black"})  # black group many libs, eye at (2,2)
json.dump(P1,open('/work/testpos4.json','w'))
print(json.dumps({"positions":P1,"activations":True}))
```