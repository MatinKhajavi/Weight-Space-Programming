
exec(open('/work/base.py').read()); exec(open('/work/l5.py').read()); exec(open('/work/l6.py').read())
W1,B1=build_L1(inp)
W2,B2=prop(rep1); W3,B3=prop(repD); W4,B4=prop(repD)
W5,B5=build_L5()
W6,B6,Wo,Bo=build_L6_out()
T={'L1.W':W1,'L1.b':B1,'L2.W':W2,'L2.b':B2,'L3.W':W3,'L3.b':B3,'L4.W':W4,'L4.b':B4,
   'L5.W':W5,'L5.b':B5,'L6.W':W6,'L6.b':B6,'out.W':Wo,'out.b':Bo}
def show_col(Wm,Bm,tgt,names):
    nz=np.nonzero(Wm[:,tgt])[0]
    print(" ".join(f"{names[int(k)//81]}{divmod(int(k)%81,9)}:{Wm[k,tgt]:+g}" for k in nz), "| bias", Bm[tgt])
