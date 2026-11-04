```python
src=open('/work/builder15.py').read()
src=src.replace("""        B[6][L6_AP+p]=prm['dens_b']*prm['dens_t']
""","""        W[6][OPP+p,L6_AP+p]=3.
        B[6][L6_AP+p]=prm['dens_b']*prm['dens_t']-2.
""")
open('/work/builder15.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v15.json'))
W,B,Wo,bo=build(prm)
print(W[6][OPP+40,L6_AP+40],B[6][L6_AP+40])
save_all(save,W,B,Wo,bo)
```