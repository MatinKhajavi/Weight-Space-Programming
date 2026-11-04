```python
src=open('/work/builder6.py').read()
src=src.replace("""                W[6][OWN+q,L6_O+p]+=-kin[dd]; W[6][OPP+q,L6_O+p]+=kin[dd]
""","")
# add SING units at L6_O block
src=src.replace("""        B[6][L6_SA+p]=2-deg
        B[6][L6_EYE+p]=-(deg-1)
""","""        B[6][L6_SA+p]=2-deg
        B[6][L6_EYE+p]=-(deg-1)
        # single opp stone in atari at p (as stone): 4*opp_p + #own/edge nbrs - 6 - 4*#opp nbrs
        W[6][OPP+p,L6_O+p]+=4
        for (rr,cc) in nb:
            q=P(rr,cc); W[6][OWN+q,L6_O+p]+=1; W[6][OPP+q,L6_O+p]+=-4
        B[6][L6_O+p]=(4-deg)-6
""")
src=src.replace("""            Wo[L6_W+q,p]+=-prm['cI']; Wo[L6_O+q,p]+=prm['cI']
""","""            Wo[L6_O+q,p]+=prm['sing']
""")
src=src.replace("""            Wo[L6_W+q,p]+=-prm['aW']*w/tot; Wo[L6_O+q,p]+=-prm['aO']*w/tot""","""            Wo[L6_W+q,p]+=-prm['aW']*w/tot""")
open('/work/builder7.py','w').write(src)
exec(src)
import json
prm=dict(cap=7.,sing=20.,esc=9.,sa=12.,eye=40.,k0=2.,k1=1.,k2=0.5,k3=0.25,aW=0.5,
         lineval=[-2.5,-0.3,1.5,1.2,1.5],tengen=1.0,**{'pass':-8.})
json.dump(prm,open('/work/prm_v7.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
print('pass bias',bo[81])
save_all(save,W,B,Wo,bo)
print('saved v7')
```