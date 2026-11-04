E6-strong scored 43.2% over H + H2 (GSM8K 52.6%, MATH 37.5%). The D-set pattern of GSM8K up and MATH down does not appear here at the pooled level. I will split the result into H and H2. On H I will compute the paired E6-strong-vs-E6 and E6-strong-vs-original differences against gen_18, by source. I will then pool E6-strong vs E6 over D and H, with per-problem clustering, to judge the strength choice. If the pooled difference is near zero, with the source split reversing between sets, the two strengths are equivalent. I would then prefer the smaller edit, E6, which also has the cleaner held-out confirmation.

```python
# Hypothesis: on held-out H the D-set source split (GSM up / MATH down) does not replicate; pooled E6s-E6 ~ 0.
G[20]=readj('/work/gen_20.jsonl')
G['20H']=[r for r in G[20] if r['id'] in set(H)]; G['20H2']=[r for r in G[20] if r['id'] in set(H2)]
print('E6s on H', summ_g(G['20H']), '| on H2', summ_g(G['20H2']))
for a,b,lab in [('18e','20H','E6s-E6 (H)'),('18o','20H','E6s-orig (H)'),('18o','18e','E6-orig (H)')]:
    print(lab, 'all', pdiff(G[a],G[b]), 'gsm', pdiff(G[a],G[b],'gsm8k'), 'math', pdiff(G[a],G[b],'math'))
def pooled(pairs_list, src=None):
    ds=[]
    for ga,gb in pairs_list:
        A={r['id']:r for r in ga}; B={r['id']:r for r in gb}
        for i in A:
            if i in B and (src is None or A[i]['source']==src):
                n_=min(len(A[i]['samples']),len(B[i]['samples']))
                ds.append(np.mean([B[i]['samples'][j]['correct']-A[i]['samples'][j]['correct'] for j in range(n_)]))
    ds=np.array(ds,float); return round(100*ds.mean(),2), round(100*ds.std(ddof=1)/np.sqrt(len(ds)),2), len(ds)
for src in [None,'gsm8k','math']:
    print('POOLED E6s-E6 (D n=2 + H n=1)', src, pooled([(G[16],G[19]),(G['18e'],G['20H'])], src))
```