```python
t7=open('/work/trunk7.py').read()
t11=t7.replace("for q in [p]+NB[p]: W[TX+q,u]-=1/16","for q in [p]+NB[p]: W[TX+q,u]-=1/1024")
assert t11!=t7
open('/work/trunk11.py','w').write(t11)
h10=open('/work/head10.py').read()
h11=h10.replace("""        u=SAVE+p; add_e(u,-12); b[u]-=11
        for q in nbp: W[AXS+q,u]+=1; W[OWN+q,u]-=12""","""        u=SAVE+p; add_e(u,1); b[u]-=5
        for q in nbp: W[LX+q,u]+=4; W[U1+q,u]-=4; W[OWN+q,u]+=1; W[U1+q,u]-=1""")
h11=h11.replace("""        for q in nbp: W[LX+q,u]+=P['w_att2']; W[U1+q,u]-=P['w_att2']""","""        for q in nbp: W[LX+q,u]+=P.get('w_att2',0); W[U1+q,u]-=P.get('w_att2',0)""")
h11=h11.replace("Wo[SAVE+p,p]=P.get('w_ko',0)","Wo[SAVE+p,p]=P['w_attu']")
assert "W[LX+q,u]+=4; W[U1+q,u]-=4; W[OWN+q,u]+=1" in h11 and "P['w_attu']" in h11
open('/work/head11.py','w').write(h11)
g=dict(save=save); exec(t11+h11,g)
import json; P=json.load(open('/work/P_v10.json')); P.update(w_attu=2, w_att2=0, w_cap=6)
g['save_all'](P); json.dump(P,open('/work/P_v11.json','w'))
open('/work/notes.txt','a').write("pass15 v10: strong 43%. Observed: after rep-fixes, nonrep winrate fell from ~60-70% (v1-v3b) to ~40%. v11: trunk11 (LX binary any size /1024), head11: ATT v3b-style unit in slot 81 (w_attu=2), no KO, w_cap=6, keep EXL & SELF(U1 trick). P_v11.json\n")
print(P)
```