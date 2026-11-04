```python
src=open('/work/builder19.py').read()
old="""            W[6][OWN+q,L6_EYE+p]+=prm['kM']; W[6][OPP+q,L6_EYE+p]+=prm['kM']
"""
assert old in src
src=src.replace(old,"""            W[6][OWN+q,L6_EYE+p]+=prm['kM']*1; W[6][OPP+q,L6_EYE+p]+=-prm['kM']*4
""")
old2="""        B[6][L6_EYE+p]=-prm['kM']*deg-7.5-prm['kT']
"""
assert old2 in src
src=src.replace(old2,"""        W[6][OPP+p,L6_EYE+p]+=prm['kM']*4
        B[6][L6_EYE+p]=prm['kM']*((4-deg)-7)-7.5-prm['kT']
""")
# output: penalty at neighbors of q instead of at p
old3="""Wo[L6_EYE+p,p]=-prm['kw']"""
assert old3 in src
src=src.replace(old3,"""pass""")
src=src.replace("""            Wo[L6_AP+q,p]+=prm['cap']""","""            Wo[L6_EYE+q,p]+=-prm['kw']
            Wo[L6_AP+q,p]+=prm['cap']""")
open('/work/builder23.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v22.json')); prm.update(kM=10.,kT=0.,kw=1.5)
json.dump(prm,open('/work/prm_v23.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
# inspect one SINGL unit
p=P(4,4)
print({int(i):float(W[6][i,L6_EYE+p]) for i in np.nonzero(W[6][:,L6_EYE+p])[0] if abs(W[6][i,L6_EYE+p])>1.5}, B[6][L6_EYE+p])
save_all(save,W,B,Wo,bo)
```