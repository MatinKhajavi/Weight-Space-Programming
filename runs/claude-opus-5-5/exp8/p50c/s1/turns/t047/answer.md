fwd_4 saved attention patterns for all six full-attention layers on both probe texts, plus MLP activations at every layer. In probe 2 the loop runs from token 1353 to 1542 with a 38-token period. Token log-probabilities rise sharply in the last quarter (mean -0.28 versus about -0.7 earlier), which is the loop becoming near-deterministic. I'll now compute, for each of the 48 full-attention heads, how much attention a loop token puts on the induction target: the token that followed its copy one cycle earlier (offset t-38+1), summed over all earlier cycles. As a baseline I'll use the same quantity on pre-loop text, where the target is the token after the most recent earlier occurrence of the same token. Heads with high in-loop induction mass are candidates for driving the self-reinforcing copy.

```python
# Hypothesis: a few full-attn heads act as induction heads that copy the previous loop cycle; quantify per head
ids2=ids4[1,:L4[1]]
def ind_targets(ids, positions):
    # for each position t, list of j = (k+1) where ids[k]==ids[t], k<t (token after earlier occurrence of current token)
    last={}; tg={}
    occ=collections.defaultdict(list)
    for t in range(len(ids)):
        if t in positions: tg[t]=[k+1 for k in occ[ids[t]] if k+1<t]
        occ[ids[t]].append(t)
    return tg
lpos=set(range(1353+per,1543)); ppos=set(range(200,1300))
tg_l=ind_targets(ids2,lpos); tg_p=ind_targets(ids2,ppos)
out=[]
for l in [3,7,11,15,19,23]:
    Aa=np.load(f'/work/fwd_4/attn_{l}.npy', mmap_mode='r')
    A2=np.asarray(Aa[1]) if Aa.ndim==4 else None
    for h in range(8):
        Ah_=A2[h]
        il=np.mean([Ah_[t,tg_l[t]].sum() for t in tg_l if tg_l[t]]); 
        ip=np.mean([Ah_[t,tg_p[t]].sum() for t in tg_p if tg_p[t]])
        prev=np.mean([Ah_[t,t-per+1] for t in lpos])
        sink=np.mean([Ah_[t,0] for t in lpos])
        out.append((l,h,round(il,3),round(ip,3),round(prev,3),round(sink,3)))
print(Aa.shape)
for o in sorted(out,key=lambda o:-o[2])[:14]: print(o)
```