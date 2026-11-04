```python
src=open('/work/builder9.py').read()
# replace OT unit definitions with A2W
src=src.replace("""            W[6][OWN+q,L6_OT+p]+=1.; W[6][OPP+q,L6_OT+p]+=-2.
""","")
src=src.replace("""        for (rr,cc) in ring2(r,c):
            W[6][OWN+P(rr,cc),L6_OT+p]+=0.5; W[6][OPP+P(rr,cc),L6_OT+p]+=-1.
""","")
src=src.replace("""        B[6][L6_OT+p]=-prm['ot_t']
""","""        W[6][OWN+p,L6_OT+p]=1
        for (rr,cc) in [(r,c)]+nb:
            W[6][S3+P(rr,cc),L6_OT+p]+=-100
""")
src=src.replace("""            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']""","""            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']-prm['def']; Wo[L6_OT+q,p]+=prm['def']""")
src=src.replace("""        Wo[L6_LIN+p,p]=1.; Wo[L6_OT+p,p]=-prm['ot']""","""        Wo[L6_LIN+p,p]=1.""")
open('/work/builder11.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v9.json')); prm['def']=2.0
json.dump(prm,open('/work/prm_v11.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
# check OT block weights now only A2W semantics
print(np.abs(W[6][:,L6_OT:L6_OT+81]).sum())
save_all(save,W,B,Wo,bo)
print(prm)
```