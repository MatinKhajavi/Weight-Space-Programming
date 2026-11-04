```python
src=open('/work/build4.py').read()
src=src.replace("""        L6.b[Lin6[p]] = C
""","""        L6.b[Lin6[p]] = C
        for s in range(81):
            if s==p: continue
            kk=kval(s//9-r0p, s%9-c0p, P['prox_opp'])
            if kk: L6.W[x5[s],Lin6[p]] += kk
            kk=kval(s//9-r0p, s%9-c0p, P['prox_own'])
            if kk: L6.W[o5[s],Lin6[p]] += kk
""")
src=src.replace("""        deg=len(NBS[p])
        for q in NBS[p]: L6.W[o5[q],eye6[p]]+=1""","""        deg=len(NBS[p]); r0p,c0p=divmod(p,9)
        for q in NBS[p]: L6.W[o5[q],eye6[p]]+=1""")
src=src.replace("prior=[-0.5,-0.2,0.0,0.1,0.3])","prior=[-0.5,-0.2,0.0,0.1,0.3], prox_opp={(0,1):0.5,(1,1):0.3}, prox_own={})")
open('/work/build4.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers])
save_all(layers,OW,Ob,save)
print('saved v4')
```