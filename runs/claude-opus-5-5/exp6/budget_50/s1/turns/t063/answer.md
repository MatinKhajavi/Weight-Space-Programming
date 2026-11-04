```python
src=open('/work/builder4.py').read()
# add corner-cycle detector units in L6 spare (486..)
src=src.replace("""    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']""","""    u=486
    for (cr,cc_) in ((0,0),(0,8),(8,0),(8,8)):
        dr=1 if cr==0 else -1; dc=1 if cc_==0 else -1
        for d1,d2 in (((dr,0),(0,dc)),((0,dc),(dr,0))):
            Cp=(cr,cc_); Bp=(cr+d1[0],cc_+d1[1]); Dp=(cr+d2[0],cc_+d2[1]); Ap=(cr+2*d1[0],cc_+2*d1[1])
            X1=(Bp[0]+d2[0],Bp[1]+d2[1]); X2=(cr+3*d1[0],cc_+3*d1[1]); X3=(Ap[0]+d2[0],Ap[1]+d2[1])
            W[6][OWN+P(*Bp),u]+=1; W[6][OWN+P(*Dp),u]+=1
            for X in (X1,X2,X3): W[6][OPP+P(*X),u]+=1
            W[6][OWN+P(*Cp),u]+=-1; W[6][OPP+P(*Cp),u]+=-1
            B[6][u]=1-5
            Wo[u,P(*Ap)]+=-prm['cyc']
            u+=1
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']""")
open('/work/builder9.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v4.json')); prm['cyc']=30.
json.dump(prm,open('/work/prm_v9.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
print(prm)
```