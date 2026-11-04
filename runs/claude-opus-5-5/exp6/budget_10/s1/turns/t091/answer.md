```python
src=open('/work/build10.py').read()
old="""        for q in NBS[p]: L6.W[o5[q],eye6[p]]+=1
        L6.W[o5[p],eye6[p]]-=B; L6.W[x5[p],eye6[p]]-=B
        for q in NBS[p]:
            others=[t for t in NBS[q] if t!=p]
            for t in others: L6.W[x5[t],eye6[p]] -= 1.0/len(others)
        L6.b[eye6[p]] = -(deg-1)"""
new="""        R2p=[q for q in range(81) if 0<abs(q//9-p//9)+abs(q%9-p%9)<=2]
        kap = 1.0 + P['mu']*(len(R2p)-1.5)
        for q in NBS[p]: L6.W[o5[q],eye6[p]]+=1+B
        L6.W[o5[p],eye6[p]]-=B; L6.W[x5[p],eye6[p]]-=B
        for q in NBS[p]:
            others=[t for t in NBS[q] if t!=p]
            for t in others: L6.W[x5[t],eye6[p]] -= kap/len(others)
        for r in R2p:
            L6.W[o5[r],eye6[p]] += P['mu']; L6.W[x5[r],eye6[p]] += P['mu']
        L6.b[eye6[p]] = -(deg-1) - B*deg"""
assert old in src
src=src.replace(old,new)
src=src.replace("P0=1.0, scale=5.0,","P0=1.0, scale=5.0, mu=1.0,")
open('/work/build11.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers])
```