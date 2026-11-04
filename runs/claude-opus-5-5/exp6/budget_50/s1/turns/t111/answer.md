```python
src=open('/work/builder15.py').read()
# 1) remove EYE weights; use L6_EYE block for KOW
src=src.replace("""            W[6][OWN+q,L6_EYE+p]+=1
""","""            W[6][OWN+q,L6_EYE+p]+=prm['kM']; W[6][OPP+q,L6_EYE+p]+=prm['kM']
""")
src=src.replace("""        B[6][L6_EYE+p]=-(deg-1)
""","""        B[6][L6_EYE+p]=-prm['kM']*deg-7.5-prm['kT']
        for (rr2,cc2) in PTS:
            W[6][OWN+P(rr2,cc2),L6_EYE+p]+=1; W[6][OPP+P(rr2,cc2),L6_EYE+p]+=-1
        W[6][486,L6_EYE+p]+=15.
""")
src=src.replace("""Wo[L6_EYE+p,p]=-prm['eye']""","""Wo[L6_EYE+p,p]=-prm['kw']""")
# OT bias deg-dependent
src=src.replace("""        B[6][L6_OT+p]=-prm['ot_t']""","""        B[6][L6_OT+p]=-prm['ot_t']+(4-deg)*prm['ot_deg']""")
# white flag carry
src=src.replace("""    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']""","""    W[1][I(0,0,2),486]=1.
    for k in (2,3,4,5): W[k][486,486]=1.
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']""")
assert src.count("prm['kM']")==3
open('/work/builder19.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v17.json')); prm.update(kM=10.,kT=5.,kw=1.0,ot_deg=1.5)
json.dump(prm,open('/work/prm_v19.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
# positions to verify: eyes (interior, edge, corner) not filled ; ko capture when ahead avoided
P2=[{"board":["X.X......","XXX......",".........",".........",".........",".........",".........",".........","........."],"to_move":"black"},
    {"board":[".X.......","XX.......",".........",".........",".........",".........",".........",".........","........."],"to_move":"black"},
    {"board":[".........",".XXX.....",".X.X.....",".XXX.....",".........",".........",".........",".........","........."],"to_move":"black"},
    # late game, black far ahead, ko capture available at (0,4)?  construct: white stone (0,4) in atari, black can capture at (0,5) into ko shape
    {"board":["XXXXOX.XX","XXXXXOXXX","XXXXXXXXX","XXXXXXXXX","XXXXXXXXX","OOOO.XXXX","OOOOOOOOO",".O.O.O.O.","OO.O.O.OO"],"to_move":"black"},
    {"board":["XXX.X.X..","XOXXXXXXX","XXXXXXXXX","XXXXXXXXX","XXXXXXXXX","OOOOOOOOO","OOOOOOOOO",".O.O.O.O.","OO.O.O.OO"],"to_move":"black"}]
print(json.dumps({"positions":P2,"activations":False}))
```