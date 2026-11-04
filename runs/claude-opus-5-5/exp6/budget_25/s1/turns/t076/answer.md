```python
h9=open('/work/head9.py').read()
h10=h9.replace("for q in nbp: W[SAFE3+q,u]-=2048; W[AXS+q,u]-=3","for q in nbp: W[SAFE3+q,u]-=2048; W[AXS+q,u]-=3; W[OWN+q,u]-=1; W[U1+q,u]+=1")
assert "W[OWN+q,u]-=1; W[U1+q,u]+=1" in h10
open('/work/head10.py','w').write(h10)
g=dict(save=save); exec(open('/work/trunk7.py').read()+h10,g)
import json; P=json.load(open('/work/P_v9.json')); P.update(w_exl=40, w_self=-12)
g['save_all'](P); json.dump(P,open('/work/P_v10.json','w'))
open('/work/notes.txt','a').write("pass14 v9: strong 40%, weak 97% (captures back, reps 4). v10: head10 SELF adds -sum(own-U1) (nonatari own nbr counts as 1 lib), w_exl=40, w_self=-12. P_v10.json\n")
print('ok')
```