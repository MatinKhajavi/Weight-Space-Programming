```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(5)
st=collections.Counter()
for gi,g in enumerate(G):
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        if la!=1 or cap>0: continue
        ch,idx=chains(b)
        e=sum(1 for q in NBR[p] if b[q]==0)
        v=collections.Counter(); capc=0; atari_adj=False
        for q in NBR[p]:
            if b[q]==netcol:
                L=len(ch[idx[q]][2]); sz=len(ch[idx[q]][1]); lq=sum(1 for y in NBR[q] if b[y]==0)
                s1 = 1 if L>=2 else 0
                if L==1: atari_adj=True
                v['A']+=s1
                v['loc']+=s1 + (1 if (L>=2 and (lq>=3 or sz>=2)) else 0)
                v['l3']+=s1 + (1 if (L>=2 and lq>=3) else 0)
            elif b[q]==-netcol and len(ch[idx[q]][2])==1 and len(ch[idx[q]][1])==1: capc+=1
        key=tuple((name, e+v[name]+2*capc<=1) for name in ('A','loc','l3'))+(('escape' if atari_adj else 'noesc'),)
        st[key]+=1
for k,vv in st.items(): print(k,vv)
```