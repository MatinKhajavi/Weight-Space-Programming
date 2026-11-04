```python
src=open('/work/builder4.py').read()
# replace LIN/OT computation in L6 with W/O influence units
old_l6=src[src.index("    # ---- L6"):src.index("def save_all")]
new_l6 = r'''    # ---- L6
    L6_W,L6_O=324,405
    kin={0:prm['k0'],1:prm['k1'],2:prm['k2'],3:prm['k3']}
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        tch = XO if par(r,c)==0 else XE
        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
        for (rr,cc) in [(r,c)]+nb:
            q=P(rr,cc)
            W[6][SW+q,L6_AW+p]+=-100; W[6][tch+q,L6_AP+p]+=-100
        for (rr,cc) in nb:
            q=P(rr,cc)
            W[6][OWN+q,L6_SA+p]+=1; W[6][OPP+q,L6_SA+p]+=1
            W[6][S3+q,L6_SA+p]+=-100
            W[6][OWN+q,L6_EYE+p]+=1
        B[6][L6_SA+p]=2-deg
        B[6][L6_EYE+p]=-(deg-1)
        # influence at p
        for (rr,cc) in PTS:
            dd=abs(rr-r)+abs(cc-c)
            if dd<=3:
                q=P(rr,cc)
                W[6][OWN+q,L6_W+p]+=kin[dd]; W[6][OPP+q,L6_W+p]+=-kin[dd]
                W[6][OWN+q,L6_O+p]+=-kin[dd]; W[6][OPP+q,L6_O+p]+=kin[dd]
    kag={0:1.0,1:0.5,2:0.25}
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']
            Wo[L6_W+q,p]+=-prm['cI']; Wo[L6_O+q,p]+=prm['cI']
        Wo[L6_SA+p,p]=-prm['sa']; Wo[L6_EYE+p,p]=-prm['eye']
        xs=[(P(rr,cc),kag[abs(rr-r)+abs(cc-c)]) for (rr,cc) in PTS if abs(rr-r)+abs(cc-c)<=2]
        tot=sum(w for _,w in xs)
        for q,w in xs:
            Wo[L6_W+q,p]+=-prm['aW']*w/tot; Wo[L6_O+q,p]+=-prm['aO']*w/tot
        bo[p]=prm['lineval'][line(r,c)]
    bo[P(4,4)]+=prm['tengen']
    bo[81]=prm['pass']
    return W,B,Wo,bo

'''
src=src.replace(old_l6,new_l6)
open('/work/builder6.py','w').write(src)
exec(src)
import json
prm=dict(cap=16.,esc=9.,sa=5.,eye=40.,k0=2.,k1=1.,k2=0.5,k3=0.25,cI=0.2,aW=0.5,aO=0.3,
         lineval=[-2.5,-0.3,1.5,1.2,1.5],tengen=1.0,**{'pass':-8.})
json.dump(prm,open('/work/prm_v6.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
print('saved v6')
```