```python
exec(open('/work/build_all.py').read())
# independent re-derivation of propagation layer weights (weight-level check only)
def nbr(r,c):
    return [(r+dr,c+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)) if 0<=r+dr<9 and 0<=c+dc<9]
idx=lambda t,r,c: t*81+r*9+c
def expected_prop(first):
    Wx=np.zeros((512,512)); Bx=np.zeros(512)
    for r in range(9):
        for c in range(9):
            Wx[idx(0,r,c),idx(0,r,c)]=1; Wx[idx(1,r,c),idx(1,r,c)]=1
            for col in (0,1):
                for sig in (0,1):
                    tgt=idx(2+2*col+sig,r,c)
                    cells=[(r,c)]+nbr(r,c)
                    for (rr,cc) in cells:
                        if first:
                            even=((rr+cc)%2==0)
                            if (even and sig==0) or ((not even) and sig==1):
                                Wx[idx(2+2*col,rr,cc),tgt]+=1
                            else:
                                Wx[idx(3+2*col,rr,cc),tgt]+=1; Wx[idx(2+2*col,rr,cc),tgt]-=1
                        else:
                            Wx[idx(2+2*col+sig,rr,cc),tgt]+=1
                    for (rr,cc) in nbr(r,c): Wx[idx(col,rr,cc),tgt]-=1
                    Wx[idx(col,r,c),tgt]+=1
                    Bx[tgt]=-1
    return Wx,Bx
e2,b2=expected_prop(True); e3,b3=expected_prop(False)
print("L2 match", np.abs(e2-W2).max(), np.abs(b2-B2).max())
print("L3 match", np.abs(e3-W3).max(), np.abs(b3-B3).max(), "L4", np.abs(e3-W4).max())
# L1 check
e1=np.zeros((243,512)); eb1=np.zeros(512)
for r in range(9):
    for c in range(9):
        e1[(r*9+c)*3+0,idx(0,r,c)]=1; e1[(r*9+c)*3+1,idx(1,r,c)]=1
        for t,pl,m in ((2,0,1),(3,0,2),(4,1,1),(5,1,2)):
            e1[(r*9+c)*3+pl,idx(t,r,c)]+=m
            for rr,cc in nbr(r,c):
                e1[(rr*9+cc)*3+0,idx(t,r,c)]+=1; e1[(rr*9+cc)*3+1,idx(t,r,c)]+=1
            eb1[idx(t,r,c)]=-len(nbr(r,c))
print("L1 match", np.abs(e1-W1).max(), np.abs(eb1-B1).max())
print("inp check", inp(2,3,1), (2*9+3)*3+1, mv(2,3))
# check loaded saved values equal
for k,v in T.items():
    print(k, np.abs(load(k)-v).max())
```