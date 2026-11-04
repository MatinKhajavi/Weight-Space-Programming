Indentation isn't the driver. In the original model, loop lines sit at the same shallow 4-space indent as the lines that close the thinking, and under A the difference is modest. I'll drop the nesting hypothesis. The remaining failure is doubt escalation: lines starting with "Wait" pile up until the model locks into re-verification. A candidate mechanism is a line-level copy: at the start of a new line, heads attend to the "Wait" tokens that began earlier lines and copy "Wait" forward, a self-reinforcing doubt habit. These could be different heads from the token-sequence induction heads I removed. I'll test this on the saved probe-2 attention: at positions whose next token is " Wait", which heads put attention on earlier " Wait" tokens, compared with positions whose next token is something else?

```python
# Hypothesis: some heads copy ' Wait' from earlier line starts (line-level doubt copying), distinct from A's heads
WAIT={13428,13784}
ids2l=list(ids2); wpos=[k for k,t in enumerate(ids2l) if t in WAIT]
qW=[t for t in range(200,len(ids2l)-1) if ids2l[t+1] in WAIT and any(k<t for k in wpos)]
qN=[t for t in range(200,len(ids2l)-1) if ids2l[t+1] not in WAIT and t%3==0]
print('n Wait tokens',len(wpos),'query pos predicting Wait',len(qW))
outW=[]
for l in [3,7,11,15,19,23]:
    Aa=np.load(f'/work/fwd_4/attn_{l}.npy',mmap_mode='r'); A2=np.asarray(Aa[1])
    for h in range(8):
        mW=np.mean([A2[h,t,[k for k in wpos if k<t]].sum() for t in qW]); mN=np.mean([A2[h,t,[k for k in wpos if k<t]].sum() for t in qN])
        outW.append((l,h,round(float(mW),3),round(float(mN),3)))
for o in sorted(outW,key=lambda o:-(o[2]-o[3]))[:10]: print(o)
```