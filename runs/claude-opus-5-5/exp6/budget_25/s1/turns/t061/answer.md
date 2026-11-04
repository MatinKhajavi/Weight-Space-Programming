```python
t6=open('/work/trunk6.py').read()
t7=t6.replace("""        for (ch,gate) in ((SO,OWN),(TX,OPP)):
            u=ch+p; W[ch+p,u]+=1""","""        for (ch,gate) in ((SO,OWN),):
            u=ch+p; W[ch+p,u]+=1""")
t7=t7.replace("for (src,dst,gate) in ((SX,VX,OPP),(TO,VO,OWN)):","for (src,dst,gate) in ((SX,VX,OPP),(TO,VO,OWN),(TX,TX,OPP)):")
t7=t7.replace("for (ch,gate,Bg) in ((SO,OWN,16384),(VO,OWN,2**23),(TX,OPP,16384),(VX,OPP,2**23)):",
              "for (ch,gate,Bg) in ((SO,OWN,16384),(VO,OWN,2**23),(TX,OPP,2**23),(VX,OPP,2**23)):")
t7=t7.replace("""        u=SAFE3+p; b[u]=-12
        for q in [p]+NB[p]: W[VO+q,u]+=1""","""        u=SAFE3+p; b[u]=-12/1024-16384; W[OWN+p,u]+=16384
        for q in [p]+NB[p]: W[VO+q,u]+=1/1024""")
t7=t7.replace("""        u=LX+p; W[OPP+p,u]+=1
        for q in [p]+NB[p]: W[TX+q,u]-=1""","""        u=LX+p; W[OPP+p,u]+=1
        for q in [p]+NB[p]: W[TX+q,u]-=1/16""")
assert "W[TX+q,u]-=1/16" in t7 and "16384; W[OWN+p,u]+=16384" in t7
open('/work/trunk7.py','w').write(t7)
h6=open('/work/head6.py').read()
h7=h6.replace("for q in nbp: W[SAFE3+q,u]-=2; W[AXS+q,u]-=3","for q in nbp: W[SAFE3+q,u]-=2048; W[AXS+q,u]-=3")
h7=h7.replace("""        u=ATT+p; b[u]-=4; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[U1+q,u]-=4""","""        u=ATT+p; b[u]-=3; add_e(u,1)
        for q in nbp: W[LX+q,u]+=2; W[U1+q,u]-=2""")
assert "W[SAFE3+q,u]-=2048" in h7 and "W[LX+q,u]+=2; W[U1+q,u]-=2" in h7
open('/work/head7.py','w').write(h7)
g=dict(save=save); exec(t7+h7,g)
import json; P=json.load(open('/work/P_v6.json'))
Ws,bs=g['build_all'](P)
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
for k in ['L1','L2','L3','L4','L5']:
    w=np.unique(Ws[k][Ws[k]!=0]); bad=[x for x in w if (np.frombuffer(np.float32(x).tobytes(),np.uint32)[0] & 0xFFFF)!=0]
    print(k,len(bad))
g['save_all'](P); json.dump(P,open('/work/P_v7.json','w'))
open('/work/notes.txt','a').write("pass10 v6: 47% BUT SAFE3 leaked onto opp stones (ungated) -> SELF broken. v7: trunk7/head7: SAFE3 gated (scaled /1024, SELF weight 2048), TX->mass channel, LX=relu(opp - v/16) graded, ATT=relu(2*opp2+e-3). P_v7.json\n")
```