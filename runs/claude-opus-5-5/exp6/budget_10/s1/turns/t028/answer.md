```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
def libs_nocap(b,p,col):
    nb=b.copy(); nb[p]=col
    ch,idx=chains(nb)
    return len(ch[idx[p]][2])
st=collections.Counter()
for k in (2,3):
    G,boards=load_pass(k)
    for gi,g in enumerate(G):
        netcol = 1 if g['net_color']=='black' else -1
        for kk,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81: continue
            b=(boards[gi,kk-1] if kk>0 else np.zeros(81)).astype(int)
            p=m['move']; la,cap=libs_after(b,p,netcol)
            ln=libs_nocap(b,p,netcol)
            if cap>0:
                st[('cap', 'nc<=1' if ln<=1 else 'nc>=2', 'size1' if cap==1 else 'size2+')]+=1
            elif la==1:
                st[('SA_nocap', 'nc<=1' if ln<=1 else 'nc>=2')]+=1
print(st)
```