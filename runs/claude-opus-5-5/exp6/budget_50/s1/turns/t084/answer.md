```python
src=open('/work/builder12.py').read()
# restore LIN: add LIN unit weights back (in L6_LIN block -> move cycle units to L6_OT block)
src=src.replace("    u=L6_LIN\n","    u=L6_OT\n")
src=src.replace("    print('cycle units',u-L6_LIN)\n","    print('cycle units',u-L6_OT)\n")
# remove A2W in OT block
src=src.replace("""        W[6][OWN+p,L6_OT+p]=1
        for (rr,cc) in [(r,c)]+nb:
            W[6][S3+P(rr,cc),L6_OT+p]+=-100
""","""        for (rr,cc) in nb:
            q=P(rr,cc)
            W[6][OPP+q,L6_LIN+p]+=prm['contact']; W[6][OWN+q,L6_LIN+p]+=prm['ownadj']
        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']
            W[6][OPP+P(rr,cc),L6_LIN+p]+=prm['oppdiag']
        B[6][L6_LIN+p]=10.
""")
src=src.replace("""Wo[L6_AW+q,p]+=prm['esc']-prm['def']; Wo[L6_OT+q,p]+=prm['def']""","""Wo[L6_AW+q,p]+=prm['esc']""")
src=src.replace("""        pass
        bo[p]=prm['lineval'][line(r,c)]""","""        Wo[L6_LIN+p,p]=1.
        bo[p]=prm['lineval'][line(r,c)]-10.""")
open('/work/builder13.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v12.json'))
json.dump(prm,open('/work/prm_v13.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
p=P(4,4); nz=np.nonzero(Wo[:,p])[0]; print([(int(i),float(Wo[i,p])) for i in nz]); print(bo[40],bo[81])
save_all(save,W,B,Wo,bo)
```