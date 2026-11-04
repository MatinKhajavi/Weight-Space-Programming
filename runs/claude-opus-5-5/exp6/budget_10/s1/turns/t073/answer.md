```python
src=open('/work/build8.py').read()
# eye gating: replace NPo-based with local opp-count based
old="""        for q in NBS[p]: L6.W[NPo[q],eye6[p]]-=1
"""
new="""        for q in NBS[p]:
            others=[t for t in NBS[q] if t!=p]
            for t in others: L6.W[x5[t],eye6[p]] -= 1.0/len(others)
"""
assert old in src
src=src.replace(old,new)
# SA2: subtract own neighbor count
old2="""            L6.W[o5[p],SAu[p]]-=B; L6.W[x5[p],SAu[p]]-=B
            L6.b[SAu[p]] = bb-deg"""
new2="""            L6.W[o5[p],SAu[p]]-=B; L6.W[x5[p],SAu[p]]-=B
            if SAu is SA2:
                for q in NBS[p]: L6.W[o5[q],SAu[p]] -= 1
            L6.b[SAu[p]] = bb-deg"""
assert old2 in src
src=src.replace(old2,new2)
src=src.replace("P0=1.0, scale=2.0,","P0=1.0, scale=3.5,")
open('/work/build9.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers], DEFAULT['scale'])
L6=layers[5]
p=79  # (8,7)
print('eye unit weights nonzero', [(int(i)//81,int(i)%81,round(float(L6.W[i,p]),2)) for i in np.nonzero(L6.W[:,p])[0]], L6.b[p])
```