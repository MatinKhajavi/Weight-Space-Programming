```python
h7=open('/work/head7.py').read()
h8=h7.replace("""        u=EYE+p; b[u]+=1-deg""","""        u=SAVE+p; add_e(u,-1)
        for q in nbp: W[AXS+q,u]+=1/12; W[OWN+q,u]-=1
        u=EYE+p; b[u]+=1-deg""")
h8=h8.replace("Wo[SELF+p,p]=P['w_self']","Wo[SELF+p,p]=P['w_self']; Wo[SAVE+p,p]=P.get('w_ko',0)")
assert "Wo[SAVE+p,p]=P.get('w_ko',0)" in h8 and "W[AXS+q,u]+=1/12" in h8
open('/work/head8.py','w').write(h8)
g=dict(save=save); exec(open('/work/trunk7.py').read()+h8,g)
import json; P=json.load(open('/work/P_v7.json')); P['w_ko']=-18; P['w_self']=-14
Ws,bs=g['build_all'](P)
print(Ws['out'][81+40,40], Ws['out'][162+40,40], P)
g['save_all'](P); json.dump(P,open('/work/P_v8.json','w'))
open('/work/notes.txt','a').write("v8: head8 adds KO unit (slot 81) = relu(sum AXS/12 - e - n_own) w_ko=-18; w_self=-14. P_v8.json. 5-cycles were ko-like recaptures.\n")
```