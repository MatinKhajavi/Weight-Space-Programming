```python
import json
exec(open('/work/goutil.py').read())
pos=json.load(open('/work/pass_1/../pass_1/inputs.npy','rb')) if False else None
E='.........'
P1=[{"board": [".........", ".O.......", ".........", "....X....", "...XO....", "....X....", ".........", ".........", "........."], "to_move": "black"}, {"board": [".........", ".........", ".........", "....O....", "...OX....", "....O....", ".........", ".......X.", "........."], "to_move": "black"}, {"board": [".OX......", "OOX......", "XX.......", ".........", ".........", ".........", ".........", ".........", "........."], "to_move": "black"}, {"board": [".........", ".........", ".........", "..XXXXX..", "..OOOOO..", "..XXXXX..", ".........", ".........", "........."], "to_move": "black"}]
# line of 4 white with libs at both ends (different parity) -> should NOT be capture
P1.append({"board": [".........", ".........", ".........", "..XXXX...", "..OOOO...", "..XXXX...", ".........", ".........", "........."], "to_move": "black"})
# long snake white chain in atari: libs only (4,1)
P1.append({"board": [".........", ".........", ".........", "XXXXXXXX.", "XOOOOOOOX", "XXXXXXXOX", "......XOX", "......XXX", "........."], "to_move": "black"})
# own territory test: black has top area
P1.append({"board": ["....X....", "....X....", "....X....", "XXXXX....", ".........", ".....OOOO", ".....O...", ".....O...", ".....O..."], "to_move": "black"})
P1.append({"moves":[]})
print(json.dumps({"positions":P1,"activations":True}))
```