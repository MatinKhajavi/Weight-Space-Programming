```python
s=open('/work/l6.py').read()
s=s.replace("set6(U(5,p), nX2+12*cap-12)","set6(U(5,p), 1-n-2*cap-2*a)                  # n=0 fill without capture/save")
s=s.replace("wTE=-20.0,wCS=0.5,","wTE=-20.0,wZ0=-2.0,")
s=s.replace("(4,PA['wTE']),(5,PA['wCS'])","(4,PA['wTE']),(5,PA['wZ0'])")
open('/work/l6.py','w').write(s)
exec(open('/work/build_all.py').read())
n5=['O','X','AO','AX','LX','IXP']
print("L6 Z0 (4,4):"); show_col(W6,B6,U(5,(4,4)),n5)
print("out col (4,4):", [(int(k)//81, float(Wo[k,P((4,4))])) for k in np.nonzero(Wo[:,P((4,4))])[0]], Bo[P((4,4))])
for k,v in T.items():
    assert v.shape==tuple(SHAPES[k]); save(k,v.astype(np.float32))
print(saved())
```