```python
src=open('/work/builder15.py').read()
src=src.replace("""    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']""","""    # white flag carry L1..L5 at unit 486 ; pass unit at L6 494
    WF=486
    W[1][I(0,0,2),WF]=1.
    for k in (2,3,4,5): W[k][WF,WF]=1.
    PU=494
    for (r,c) in PTS:
        p=P(r,c)
        W[6][OWN+p,PU]+=1+prm['late_w']; W[6][OPP+p,PU]+=-1+prm['late_w']
    W[6][WF,PU]+=15.
    B[6][PU]=-7.5-prm['late_w']*prm['late_n']-prm['lead_T']
    Wo[PU,81]+=prm['pass_w']
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']""")
open('/work/builder16.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v15.json')); prm.update(late_w=0.3,late_n=55.,lead_T=10.,pass_w=2.)
json.dump(prm,open('/work/prm_v16.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
```