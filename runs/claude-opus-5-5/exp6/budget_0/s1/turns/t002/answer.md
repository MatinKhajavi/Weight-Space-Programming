```python
def show_col(Wm,Bm,tgt,names):
    nz=np.nonzero(Wm[:,tgt])[0]
    items=[]
    for k in nz:
        t,pp=divmod(int(k),81)
        items.append(f"{names[t]}{divmod(pp,9)}:{Wm[k,tgt]:+g}")
    print(" ".join(items), "| bias", Bm[tgt])
n1=['O','X','aO','bO','aX','bX']; nk=['O','X','WxO','WyO','WxX','WyX']
p=(4,4); q=(4,5)
print("L2 WxO at (4,4) even:"); show_col(W2,B2,U(2,p),n1)
print("L2 WyO at (4,4) even:"); show_col(W2,B2,U(3,p),n1)
print("L2 WxX at (4,5) odd:"); show_col(W2,B2,U(4,q),n1)
print("L3 WyX at (0,0):"); show_col(W3,B3,U(5,(0,0)),nk)
print("L6 Lin (4,4):"); show_col(W6,B6,U(0,p),nk)
print("L6 SA (4,4):"); show_col(W6,B6,U(1,p),nk)
print("L6 E2 (0,0):"); show_col(W6,B6,U(2,(0,0)),nk)
print("L6 Cap (0,0):"); show_col(W6,B6,U(5,(0,0)),nk)
print("L1 aO (0,0):", [ (i, W1[i,U(2,(0,0))]) for i in np.nonzero(W1[:,U(2,(0,0))])[0]], B1[U(2,(0,0))])
```