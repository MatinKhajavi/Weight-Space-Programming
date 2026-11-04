```python
h4=open('/work/head4.py').read()
h5=h4.replace("LIN=0;SAVE=81;SELF=162;EYE=243;INT=324;C6=405","LIN=0;SAVE=81;SELF=162;ATT=243;EYE=324;INT=405")
h5=h5.replace("W,b=Z(); b[C6]=1","W,b=Z()")
h5=h5.replace("for q in nbp: W[AX+q,u]+=P['w_cap']; W[AXS+q,u]+=P['w_capS']","for q in nbp: W[U1+q,u]+=P['w_cap']; W[AXS+q,u]+=P['w_capS']")
h5=h5.replace("""        for q in nbp: W[AO+q,u]+=8; W[OWN+q,u]+=2; W[LO+q,u]-=2; W[LO+q,u]+=1; W[AO+q,u]-=1""",
"""        for q in nbp: W[U1+q,u]+=8; W[OWN+q,u]+=2; W[LO+q,u]-=2""")
h5=h5.replace("""        u=EYE+p; b[u]+=1-deg""","""        u=ATT+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[U1+q,u]-=4; W[OWN+q,u]+=1; W[LO+q,u]-=1
        u=EYE+p; b[u]+=1-deg""")
h5=h5.replace("""Wo[SAVE+p,p]=P['w_save']; Wo[SELF+p,p]=P['w_self']""","""Wo[SAVE+p,p]=P['w_save']; Wo[SELF+p,p]=P['w_self']; Wo[ATT+p,p]=P['w_att']""")
h5=h5.replace("P4=dict(","P5=dict(w_att=2,").replace("P4['pass']=-3","P5['pass']=-3")
open('/work/head5.py','w').write(h5)
g=dict(save=save); exec(open('/work/trunk5.py').read()+h5,g)
P=g['P5']; print(P)
Ws,bs=g['build_all'](P)
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
g['save_all'](P)
import json; json.dump(P,open('/work/P_v5.json','w'))
open('/work/notes.txt','a').write("pass6 v4: strong 38.5% (no ATT hurt), weak 94%. v5: trunk5.py+head5.py: L5=own,opp,U1(AO+AX mixed),AXS,LO,LX; ATT back. P_v5.json\n")
```