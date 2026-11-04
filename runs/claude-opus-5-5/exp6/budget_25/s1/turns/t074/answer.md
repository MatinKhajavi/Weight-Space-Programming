```python
h8=open('/work/head8.py').read()
h9=h8.replace("""        u=SAVE+p; add_e(u,-1)
        for q in nbp: W[AXS+q,u]+=1/12; W[OWN+q,u]-=1""","""        u=SAVE+p; add_e(u,-12); b[u]-=11
        for q in nbp: W[AXS+q,u]+=1; W[OWN+q,u]-=12""")
h9=h9.replace("""        u=ATT+p; b[u]-=3; add_e(u,1)
        for q in nbp: W[LX+q,u]+=2; W[U1+q,u]-=2""","""        u=ATT+p
        for q in nbp: W[U1+q,u]+=1; W[OWN+q,u]-=2
        u=LIN+p
        for q in nbp: W[LX+q,u]+=P['w_att2']; W[U1+q,u]-=P['w_att2']""")
h9=h9.replace("Wo[ATT+p,p]=P['w_att']","Wo[ATT+p,p]=P['w_exl']")
assert "W[OWN+q,u]-=12" in h9 and "P['w_exl']" in h9 and "P['w_att2']" in h9
open('/work/head9.py','w').write(h9)
g=dict(save=save); exec(open('/work/trunk7.py').read()+h9,g)
import json; P=json.load(open('/work/P_v8.json')); P.update(w_att2=3, w_cap=9, w_exl=24, w_ko=-18)
Ws,bs=g['build_all'](P)
# sanity on LIN K positivity: min possible arg roughly
print(P)
g['save_all'](P); json.dump(P,open('/work/P_v9.json','w'))
open('/work/notes.txt','a').write("pass13 v8: strong 41% captures collapsed (KO fired on dominoes; big captures blocked by SELF). v9: head9: KO=relu(sumAXS-11-12(e+nown)); ATT folded linear into LIN (w_att2*sum(LX-U1)); EXL unit (slot 243)=relu(sumU1-2nown) w_exl=24 (lone big captures exempt). P_v9.json\n")
```