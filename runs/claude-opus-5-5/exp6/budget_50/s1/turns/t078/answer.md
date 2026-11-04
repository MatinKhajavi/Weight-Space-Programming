```python
src=open('/work/builder11.py').read()
# remove LIN unit weights
src=src.replace("""            W[6][OPP+q,L6_LIN+p]+=prm['contact']; W[6][OWN+q,L6_LIN+p]+=prm['ownadj']
""","")
src=src.replace("""        for (rr,cc) in diags(r,c):
            W[6][OWN+P(rr,cc),L6_LIN+p]+=prm['owndiag']
            W[6][OPP+P(rr,cc),L6_LIN+p]+=prm['oppdiag']
""","")
src=src.replace("""        B[6][L6_LIN+p]=10.
""","")
src=src.replace("""        Wo[L6_LIN+p,p]=1.""","""        pass""")
src=src.replace("""        bo[p]=prm['lineval'][line(r,c)]-10.""","""        bo[p]=prm['lineval'][line(r,c)]""")
# replace corner units with edge-cycle units placed at L6_LIN block
old=src[src.index("    u=486"):src.index("    for (r,c) in PTS:\n        p=P(r,c); nb=nbrs(r,c)\n        for (rr,cc) in nb:\n            q=P(rr,cc)\n            Wo[L6_AP+q,p]")]
new='''    u=L6_LIN
    def onb(x): return 0<=x[0]<N and 0<=x[1]<N
    for side in range(4):
        for t in range(1,8):
            if side==0: Bp=(0,t); e=(0,1); n_=(1,0)
            elif side==1: Bp=(8,t); e=(0,1); n_=(-1,0)
            elif side==2: Bp=(t,0); e=(1,0); n_=(0,1)
            else: Bp=(t,8); e=(1,0); n_=(0,-1)
            for s in (1,-1):
                Ap=(Bp[0]+s*e[0],Bp[1]+s*e[1]); Cp=(Bp[0]-s*e[0],Bp[1]-s*e[1])
                lits=[]  # (index, weight) ; count of literals
                W[6][OWN+P(*Bp),u]+=1; cnt=1
                W[6][OWN+P(*Cp),u]+=-1; W[6][OPP+P(*Cp),u]+=-1; cnt+=1; extra=1  # empty_C = 1-own-opp
                for X in ((Cp[0]-s*e[0],Cp[1]-s*e[1]),(Cp[0]+n_[0],Cp[1]+n_[1])):
                    if onb(X): W[6][OWN+P(*X),u]+=1; cnt+=1
                for X in ((Ap[0]+s*e[0],Ap[1]+s*e[1]),(Ap[0]+n_[0],Ap[1]+n_[1]),(Bp[0]+n_[0],Bp[1]+n_[1])):
                    if onb(X): W[6][OPP+P(*X),u]+=1; cnt+=1
                B[6][u]=extra-(cnt-1)
                Wo[u,P(*Ap)]+=-prm['cyc']
                u+=1
    print('cycle units',u-L6_LIN)
'''
src=src.replace(old,new)
open('/work/builder12.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v11.json'))
json.dump(prm,open('/work/prm_v12.json','w'))
W,B,Wo,bo=build(prm)
for k in range(1,7): print(k,len(np.where(np.abs(W[k]).sum(0)>0)[0]))
save_all(save,W,B,Wo,bo)
# sanity: verify the cycle-pattern units on the earlier trigger boards via Go-logic (pattern match only)
```