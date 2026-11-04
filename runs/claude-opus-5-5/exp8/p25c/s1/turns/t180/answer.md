The token layout is clear: prompt, then a 120-token lead, then 6 loop periods of 34 tokens (text 4) or 40 tokens (text 13). I can now profile every L23 head by region: prompt, lead (normal reasoning just before the loop), and loop periods k = 1 to 6. For each region I'll measure the attention share on the sink, on the token itself, and on earlier copies of the same token. Inside the loop I'll also measure the share on the aligned position one period back (t − P, "same") and on the position just after it (t − P + 1, "induction").

For h2 to be a credible anti-repetition target, its same-token share should be low in prompt/lead and rise across loop periods. For contrast, h5 and h6 should show the induction and same-token rise I recorded before.

```python
# Per-head L23 attention profile by region (prompt / lead / loop period k): sink, self, same-token(any earlier), aligned prev period (t-P), induction (t-P+1)
def prof(b, P, n_lead=120):
    T=lens5[b]; ids=ids5[b,:T]; n_body=6*P; s_lead=T-n_body-n_lead; s_body=T-n_body
    regs={'prompt':range(1,s_lead),'lead':range(s_lead,s_body)}
    for k in range(1,7): regs[f'k{k}']=range(s_body+(k-1)*P, s_body+k*P)
    out={}
    for h in range(8):
        Ah=A23[b,h]; row={}
        for nm,rg in regs.items():
            sk=sl=sm=al=ind=0
            for t in rg:
                a=Ah[t,:t+1]; sk+=a[0]; sl+=a[t]
                same=np.where(ids[:t]==ids[t])[0]; sm+=a[same].sum() if len(same) else 0
                if t-P>=0: al+=a[t-P]
                if t-P+1>=0 and t-P+1<t: ind+=a[t-P+1]
            n=len(rg); row[nm]=(sk/n,sl/n,sm/n,al/n,ind/n)
        out[h]=row
    return out
pf=[prof(0,34), prof(1,40)]
for h in range(8):
    s=[]
    for nm in ['prompt','lead','k1','k3','k6']:
        v=np.mean([pf[b][h][nm] for b in range(2)],axis=0)
        s.append(f"{nm}: sk{v[0]:.2f} sf{v[1]:.2f} sm{v[2]:.2f}" + (f" al{v[3]:.2f} in{v[4]:.2f}" if nm.startswith('k') else ''))
    print(f'h{h} | '+' | '.join(s))
```