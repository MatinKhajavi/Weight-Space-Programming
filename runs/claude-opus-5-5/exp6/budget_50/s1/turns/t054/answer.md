```python
src=open('/work/builder4.py').read()
old_l6=src[src.index("    # ---- L6"):src.index("def save_all")]
new_l6 = r'''    # ---- L6
    L6_U2,L6_SING=324,405
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c); deg=len(nb)
        tch = XO if par(r,c)==0 else XE
        W[6][OWN+p,L6_AW+p]=1; W[6][OPP+p,L6_AP+p]=1
        for (rr,cc) in [(r,c)]+nb:
            q=P(rr,cc)
            W[6][SW+q,L6_AW+p]+=-100; W[6][tch+q,L6_AP+p]+=-100
        for (rr,cc) in nb:
            q=P(rr,cc)
            # U1 = relu(2 - ne - 100*sum S3own)
            W[6][OWN+q,L6_SA+p]+=1; W[6][OPP+q,L6_SA+p]+=1
            W[6][S3+q,L6_SA+p]+=-100
            # U2 = relu(1 - ne - n_own) = relu(1 - deg + n_opp ... ) : 1-ne-nown = 1-deg+nown+nopp-nown = 1-deg+nopp
            W[6][OPP+q,L6_U2+p]+=1
            W[6][OWN+q,L6_EYE+p]+=1
            W[6][OWN+q,L6_SING+p]+=1; W[6][OPP+q,L6_SING+p]+=-4
        W[6][OPP+p,L6_SING+p]+=4
        B[6][L6_SA+p]=2-deg
        B[6][L6_U2+p]=1-deg
        B[6][L6_EYE+p]=-(deg-1)
        B[6][L6_SING+p]=(4-deg)-6
    for (r,c) in PTS:
        p=P(r,c); nb=nbrs(r,c)
        for (rr,cc) in nb:
            q=P(rr,cc)
            Wo[L6_AP+q,p]+=prm['cap']; Wo[L6_AW+q,p]+=prm['esc']; Wo[L6_SING+q,p]+=prm['sing']
        Wo[L6_SA+p,p]=-prm['sa']; Wo[L6_U2+p,p]=2*prm['sa']; Wo[L6_EYE+p,p]=-prm['eye']
        bo[p]=prm['lineval'][line(r,c)]
    bo[P(4,4)]+=prm['tengen']
    bo[81]=prm['pass']
    return W,B,Wo,bo

'''
src=src.replace(old_l6,new_l6)
open('/work/builder8.py','w').write(src)
exec(src)
import json
prm=dict(cap=8.,sing=20.,esc=8.,sa=9.,eye=40.,lineval=[-2.5,-0.3,1.5,1.2,1.5],tengen=1.0,**{'pass':-10.})
json.dump(prm,open('/work/prm_v8.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
print('saved v8')
```