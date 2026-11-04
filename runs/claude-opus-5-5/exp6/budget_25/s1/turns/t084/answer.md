```python
h11=open('/work/head11.py').read()
h12=h11.replace("LIN=0;SAVE=81;SELF=162;ATT=243;EYE=324;INT=405","LIN=0;SAVE=81;SELF=162;ATT=243;EYE=324;INT=405;KOU=405")
# EXL exclude singles
h12=h12.replace("""        u=ATT+p
        for q in nbp: W[U1+q,u]+=1; W[OWN+q,u]-=2""","""        u=ATT+p
        for q in nbp: W[U1+q,u]+=1; W[OWN+q,u]-=2; W[AXS+q,u]-=1/6""")
# merge EYE+INT into TERR at EYE slot; KO at INT slot
old_eye_int=h12[h12.index("        u=EYE+p; b[u]+=1-deg"):h12.index("    Wo=np.zeros")]
new_eye_int='''        u=EYE+p; b[u]+=P['terr_A']*(1-deg)-P['int_T']
        for q in nbp: W[OWN+q,u]+=P['terr_A']+P['int_own']
        for q in range(81):
            W[OWN+q,u]+=P['int_N']; W[OPP+q,u]+=P['int_N']
            if md(p,q)<=2: W[OPP+q,u]-=P['int_M']
        u=KOU+p; add_e(u,-12); b[u]-=11
        for q in nbp: W[AXS+q,u]+=1; W[OWN+q,u]-=12
'''
h12=h12.replace(old_eye_int,new_eye_int)
h12=h12.replace("Wo[EYE+p,p]=P['w_eye']; Wo[INT+p,p]=P['w_int']","Wo[EYE+p,p]=P['w_terr']; Wo[KOU+p,p]=P['w_ko']")
assert "P['w_terr']" in h12 and "W[AXS+q,u]-=1/6" in h12 and "terr_A" in h12
open('/work/head12.py','w').write(h12)
g=dict(save=save); exec(open('/work/trunk11.py').read()+h12,g)
import json; P=json.load(open('/work/P_v11.json'))
P.update(terr_A=15,int_M=3,int_own=1,int_N=0.05,int_T=2,w_terr=-2.5,w_ko=-18,w_self=-16,w_exl=55,scale=2.0)
Ws,bs=g['build_all'](P)
g['save_all'](P); json.dump(P,open('/work/P_v12.json','w'))
open('/work/notes.txt','a').write("pass16 v11: strong 56% (ATT v3b-style helps), 15 rep losses (6 ko-5cycles,6 3cycles sampled vs pass,3 6cycles). v12: head12: TERR=EYE+INT merged (slot 324), KO unit (slot 405), EXL excludes singles, w_self -16, w_exl 55, scale 2.0. P_v12.json\n")
print(P)
```