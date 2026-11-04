```python
src=open('/work/build4.py').read()
# replace L5 section
start=src.index('    # L5'); end=src.index('    # L6')
newL5='''    # L5
    L5 = Layer(512)
    o4,x4,N4 = prev['o'],prev['x'],prev['NP']
    o5=L5.alloc(81); x5=L5.alloc(81); NPo=L5.alloc(81); NPx=L5.alloc(81); sx=L5.alloc(81); S=L5.alloc(81)
    for q in range(81):
        L5.W[o4[q],o5[q]]=1; L5.W[x4[q],x5[q]]=1
        sg=1-PAR[q]; deg=len(NBS[q])
        Uo=N4[('o',sg)]; Ux=N4[('x',sg)]
        for U,Un,c in ((Uo,NPo,o4),(Ux,NPx,x4)):
            L5.W[U[q],Un[q]]+=1
            for t in NBS[q]:
                L5.W[c[t],Un[q]]-=1; L5.W[U[t],Un[q]]+=1
        # sx local: relu(2x(q) - l(q) - 2 sum x(t)), l = deg - sum(o+x)
        L5.W[x4[q],sx[q]]+=2
        for t in NBS[q]:
            L5.W[o4[t],sx[q]]+=1; L5.W[x4[t],sx[q]]+=1-2
        L5.b[sx[q]]=-deg
        # S: relu(1 + (l-2) + 3 sum own(t) - B(1-own(q)) - B*NP4o_rel(q))
        for t in NBS[q]:
            L5.W[o4[t],S[q]] += -1 + 3
            L5.W[x4[t],S[q]] += -1
        L5.W[o4[q],S[q]] += B
        L5.W[Uo[q],S[q]] -= B
        L5.b[S[q]] = 1 + deg - 2 - B
    layers.append(L5)
'''
src=src[:start]+newL5+src[end:]
# L6 modifications: eye gating, SA, Lin, tents
src=src.replace("""        L6.W[o5[p],eye6[p]]-=B; L6.W[x5[p],eye6[p]]-=B
        L6.b[eye6[p]] = -(deg-1)""","""        L6.W[o5[p],eye6[p]]-=B; L6.W[x5[p],eye6[p]]-=B
        for q in NBS[p]: L6.W[NPo[q],eye6[p]]-=1
        L6.b[eye6[p]] = -(deg-1)""")
src=src.replace("""        for q in NBS[p]:
            L6.W[x5[q],SA6[p]] += 1
            L6.W[NPo[q],SA6[p]] += 1
            L6.W[sx[q],SA6[p]] -= 2""","""        for q in NBS[p]:
            L6.W[x5[q],SA6[p]] += 1
            L6.W[o5[q],SA6[p]] += 1
            L6.W[S[q],SA6[p]] -= 1
            L6.W[sx[q],SA6[p]] -= 2""")
src=src.replace("""            L6.W[sx[q],Lin6[p]] += P['Wc']
            L6.W[mx[q],Lin6[p]] += P['Wm']
            L6.W[NPo[q],Lin6[p]] += P['We']
            L6.W[a2[q],Lin6[p]] += P['Wa']""","""            L6.W[sx[q],Lin6[p]] += P['Wc'] - P['Wm']
            L6.W[NPx[q],Lin6[p]] += P['Wm']
            L6.W[NPo[q],Lin6[p]] += P['We']""")
src=src.replace("""                    L6.W[sx[s],t[p]] += al*kw
                    L6.W[mx[s],t[p]] += 0.5*al*kw""","""                    L6.W[NPx[s],t[p]] += al*kw""")
src=src.replace("Wc=14.0, Wm=8.0, We=9.0, Wa=3.0, Weye=40.0, Wsa=25.0","Wc=14.0, Wm=12.0, We=9.0, Weye=40.0, Wsa=40.0")
open('/work/build5.py','w').write(src)
exec(src)
layers,OW,Ob=build(DEFAULT)
print([L.n for L in layers])
```