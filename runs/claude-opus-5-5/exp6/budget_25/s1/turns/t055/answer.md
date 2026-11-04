```python
t5=open('/work/trunk5.py').read()
t6=t5.replace("U1=162; AXS=243; LO=324; LX=405          # L5","U1=162; AXS=243; LO=324; LX=405; SAFE3=324; VO=243")
# L3: replace TO hop with VO mass channel
t6=t6.replace("""        for (ch,gate) in ((SO,OWN),(TO,OWN),(TX,OPP)):
            u=ch+p; W[ch+p,u]+=1
            for r in NB[p]: W[ch+r,u]+=1
            W[gate+p,u]+=2048; b[u]=-2048
        u=VX+p; W[SX+p,u]+=M
        for r in NB[p]: W[SX+r,u]+=M; W[OPP+r,u]+=1
        W[OPP+p,u]+=2**20; b[u]=-2**20+1""","""        for (ch,gate) in ((SO,OWN),(TX,OPP)):
            u=ch+p; W[ch+p,u]+=1
            for r in NB[p]: W[ch+r,u]+=1
            W[gate+p,u]+=2048; b[u]=-2048
        for (src,dst,gate) in ((SX,VX,OPP),(TO,VO,OWN)):
            u=dst+p; W[src+p,u]+=M
            for r in NB[p]: W[src+r,u]+=M; W[gate+r,u]+=1
            W[gate+p,u]+=2**20; b[u]=-2**20+1""")
t6=t6.replace("for (ch,gate,Bg) in ((SO,OWN,16384),(TO,OWN,16384),(TX,OPP,16384),(VX,OPP,2**23)):",
              "for (ch,gate,Bg) in ((SO,OWN,16384),(VO,OWN,2**23),(TX,OPP,16384),(VX,OPP,2**23)):")
t6=t6.replace("""        u=LO+p; W[OWN+p,u]+=1
        for q in [p]+NB[p]: W[TO+q,u]-=1""","""        u=SAFE3+p; b[u]=-12
        for q in [p]+NB[p]: W[VO+q,u]+=1""")
assert t6.count('VO')>=4
open('/work/trunk6.py','w').write(t6)
h5=open('/work/head5.py').read()
h6=h5.replace("for q in nbp: W[OWN+q,u]-=2; W[LO+q,u]+=2; W[AXS+q,u]-=3","for q in nbp: W[SAFE3+q,u]-=2; W[AXS+q,u]-=3")
h6=h6.replace("""        u=ATT+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[U1+q,u]-=4; W[OWN+q,u]+=1; W[LO+q,u]-=1""","""        u=ATT+p; b[u]-=4; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[U1+q,u]-=4""")
h6=h6.replace("""        u=SAVE+p; b[u]-=9; add_e(u,1)
        for q in nbp: W[U1+q,u]+=8; W[OWN+q,u]+=2; W[LO+q,u]-=2
""","")
assert 'LO+' not in h6, [l for l in h6.split('\n') if 'LO+' in l]
open('/work/head6.py','w').write(h6)
g=dict(save=save); exec(t6+h6,g)
import json; P=json.load(open('/work/P_v5b.json'))
Ws,bs=g['build_all'](P)
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
for k in Ws:
    w=np.unique(Ws[k][Ws[k]!=0]); bad=[x for x in w if (np.frombuffer(np.float32(x).tobytes(),np.uint32)[0] & 0xFFFF)!=0]
    if k!='out' and k!='L6': print(k,len(bad))
g['save_all'](P); json.dump(P,open('/work/P_v6.json','w'))
open('/work/notes.txt','a').write("pass9 v5b: strong 48%, weak 91%, many passes (false LO for big own groups blocks moves). v6: trunk6/head6: VO mass for own T -> SAFE3 (big groups optimistic); SELF uses SAFE3; ATT uses e only; no SAVE. P_v6.json\n")
```