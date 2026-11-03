```python
code5 = r'''
def build_L5():
    Wm=np.zeros((H,H),np.float32); Bm=np.zeros(H,np.float32)
    def setu(tgt,ex):
        for k,v in ex.d.items(): Wm[k,tgt]+=v
        Bm[tgt]+=ex.c
    for p in pts:
        setu(U(0,p),V(U(0,p)))
        setu(U(1,p),V(U(1,p)))
        sig = 1 if par(p)==0 else 0
        for col in (0,1):
            ex=repD(col,sig,p)+V(U(col,p))-1
            for q in NB[p]:
                ex=ex+repD(col,sig,q)-V(U(col,q))
            setu(U(2+col,p),ex)
        deg=len(NB[p])
        ex=E(c=-deg)+3*V(U(1,p))            # lone opp stone, <=2 libs
        for q in NB[p]: ex=ex+V(U(0,q))-2*V(U(1,q))
        setu(U(4,p),ex)
        ex=E(c=-deg)+3*V(U(0,p))            # lone own stone, <=2 libs
        for q in NB[p]: ex=ex+V(U(1,q))-2*V(U(0,q))
        setu(U(5,p),ex)
    return Wm,Bm
'''
open('/work/l5.py','w').write(code5)
s=open('/work/l6.py').read()
s=s.replace("watk=1.5,","watk=1.5,wlo=0.5,").replace("wsave=11","wsave=12")
s=s.replace("ring={0:-2.5,1:-1.0,2:0.3,3:0.8,4:1.5}","ring={0:-3.0,1:-1.5,2:0.3,3:0.8,4:2.0}")
s=s.replace("atk=Esum(V(U(4,q)) for q in nb)","atk=Esum(V(U(4,q)) for q in nb); lo=Esum(V(U(5,q)) for q in nb)")
s=s.replace("+PA['watk']*atk)","+PA['watk']*atk+PA['wlo']*lo)")
open('/work/l6.py','w').write(s)
print(s[:600])
exec(open('/work/base.py').read()); exec(open('/work/l5.py').read()); exec(open('/work/l6.py').read())
W1,B1=build_L1(inp)
W2,B2=prop(rep1); W3,B3=prop(repD); W4,B4=prop(repD)
W5,B5=build_L5()
W6,B6,Wo,Bo=build_L6_out()
def show_col(Wm,Bm,tgt,names):
    nz=np.nonzero(Wm[:,tgt])[0]
    print(" ".join(f"{names[int(k)//81]}{divmod(int(k)%81,9)}:{Wm[k,tgt]:+g}" for k in nz), "| bias", Bm[tgt])
nk=['O','X','WxO','WyO','WxX','WyX']; n5=['O','X','AO','AX','LX','LO']
print("L5 LO (8,4):"); show_col(W5,B5,U(5,(8,4)),nk)
print("L6 Lin (1,1):"); show_col(W6,B6,U(0,(1,1)),n5)
T={'L1.W':W1,'L1.b':B1,'L2.W':W2,'L2.b':B2,'L3.W':W3,'L3.b':B3,'L4.W':W4,'L4.b':B4,
   'L5.W':W5,'L5.b':B5,'L6.W':W6,'L6.b':B6,'out.W':Wo,'out.b':Bo}
for k,v in T.items():
    assert v.shape==tuple(SHAPES[k]); save(k,v.astype(np.float32))
print(saved(), Bo[:10], Bo[81])
```