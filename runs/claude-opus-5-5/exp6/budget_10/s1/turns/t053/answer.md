```python
src=open('/work/build5.py').read()
# L5: replace S with ext
old_S = src[src.index("        # S: relu(1 + (l-2)"):src.index("    layers.append(L5)")]
new_ext = """        # ext: relu(l(q) - 2 - B(1-own(q)))
        for t in NBS[q]:
            L5.W[o4[t],S[q]] -= 1; L5.W[x4[t],S[q]] -= 1
        L5.W[o4[q],S[q]] += B
        L5.b[S[q]] = deg - 2 - B
"""
src=src.replace(old_S,new_ext)
# SA: -A = -e - sum(own - NPo) - sum ext - 2 sum sx  ;  -e = -deg + sum(o+x)
src=src.replace("""        for q in NBS[p]:
            L6.W[x5[q],SA6[p]] += 1
            L6.W[o5[q],SA6[p]] += 1
            L6.W[S[q],SA6[p]] -= 1
            L6.W[sx[q],SA6[p]] -= 2
        L6.W[o5[p],SA6[p]]-=B; L6.W[x5[p],SA6[p]]-=B
        L6.b[SA6[p]] = 2-deg""","""        for SAu,bb in ((SA6,2),(SA2,1)):
            for q in NBS[p]:
                L6.W[x5[q],SAu[p]] += 1
                L6.W[NPo[q],SAu[p]] += 1
                L6.W[S[q],SAu[p]] -= 1
                L6.W[sx[q],SAu[p]] -= 2
            L6.W[o5[p],SAu[p]]-=B; L6.W[x5[p],SAu[p]]-=B
            L6.b[SAu[p]] = bb-deg""")
# allocation: add SA2, t3 only for line>=2
src=src.replace("""    eye6=L6.alloc(81); SA6=L6.alloc(81); Lin6=L6.alloc(81); t1=L6.alloc(81); t2=L6.alloc(81); t3=L6.alloc(81); E6=L6.alloc(1)[0]""",
"""    eye6=L6.alloc(81); SA6=L6.alloc(81); SA2=L6.alloc(81); Lin6=L6.alloc(81); t1=L6.alloc(81); t2=L6.alloc(81)
    inner=[p for p in range(81) if LINE[p]>=2]
    t3a=L6.alloc(len(inner)); t3={p:t3a[i] for i,p in enumerate(inner)}
    E6=L6.alloc(1)[0]""")
# tent weights loops: t3 is dict now -> guard
src=src.replace("""                for t in (t1,t2,t3):
                    L6.W[o5[s],t[p]] += kw; L6.W[x5[s],t[p]] -= kw
                    L6.W[NPo[s],t[p]] -= al*kw
                    L6.W[NPx[s],t[p]] += al*kw""","""                for t in ((t1,t2,t3) if p in t3 else (t1,t2)):
                    L6.W[o5[s],t[p]] += kw; L6.W[x5[s],t[p]] -= kw
                    L6.W[NPo[s],t[p]] -= al*kw
                    L6.W[NPx[s],t[p]] += al*kw""")
src=src.replace("""        for t,off in ((t1,1),(t2,0),(t3,-1)):
            L6.W[o5[p],t[p]] -= B; L6.W[x5[p],t[p]] -= B
            L6.b[t[p]] = off""","""        for t,off in (((t1,1),(t2,0),(t3,-1)) if p in t3 else ((t1,1),(t2,0))):
            L6.W[o5[p],t[p]] -= B; L6.W[x5[p],t[p]] -= B
            L6.b[t[p]] = off""")
src=src.replace("""        OW[eye6[p],p]-=P['Weye']; OW[SA6[p],p]-=P['Wsa']""","""        OW[eye6[p],p]-=P['Weye']; OW[SA6[p],p]-=P['Wsa']; OW[SA2[p],p]+=2*P['Wsa']""")
src=src.replace("""                OW[t1[r],p]+=P['Wt']*k; OW[t2[r],p]-=2*P['Wt']*k; OW[t3[r],p]+=P['Wt']*k""","""                OW[t1[r],p]+=P['Wt']*k; OW[t2[r],p]-=2*P['Wt']*k
                if r in t3: OW[t3[r],p]+=P['Wt']*k""")
open('/work/build6.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers])
```