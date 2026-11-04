```python
src=open('/work/builder9.py').read()
src=src.replace("""        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
""","""        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
        for (rr,cc) in PTS:
            dd=abs(rr-r)+abs(cc-c)
            if 0<dd<=2: W[6][OPP+P(rr,cc),L6_AP+p]+=-prm['dens_b']
        B[6][L6_AP+p]=prm['dens_b']*prm['dens_t']
""",1)
assert "dens_b" in src
open('/work/builder15.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v14.json')); prm['dens_b']=0.15; prm['dens_t']=3.0
json.dump(prm,open('/work/prm_v15.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```