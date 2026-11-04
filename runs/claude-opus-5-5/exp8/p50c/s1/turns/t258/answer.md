The full run is done: 504 generations, 76.6% accuracy and 12.3% cut-offs on these problems, which are moderately easy by selection (MATH levels 1–4, discordant openings). Now the key analysis. First, check that samples 1–2 reproduce the screened openings, confirming that seeds are fixed per problem and sample index. Then estimate the effect of the opening mode within problems. For each problem with both styles I take the difference in accuracy and cut rate between "my thought" and "a thinking" samples, average with weights n_my·n_a/(n_my+n_a), and bootstrap over problems for the standard error. A clearly positive within-problem accuracy difference confirms that the opening mode causally drives success.

```python
# Within-problem randomized test: does the opening mode (my_thought vs a_thinking) change accuracy/cut/length?
R24 = readj('/work/gen_24.jsonl')
rep = sum(sty(r['samples'][j]['text'].lstrip()[:28]) == st23[r['id']][j] for r in R24 for j in range(2)); print('screen openings reproduced:', rep, 'of', 2*len(R24))
rows=[]
for r in R24:
    for j,s in enumerate(r['samples']):
        rows.append(dict(id=r['id'], src=r['source'], j=j, sty=sty(s['text'].lstrip()[:28]), ok=s['correct'], cut=s['cut'], tok=s['tokens']))
d24 = pd.DataFrame(rows)
print(d24.groupby('sty').agg(n=('ok','size'), acc=('ok','mean'), cut=('cut','mean'), tok=('tok','median')).round(3))
def within(df, col):
    per=[]
    for i,g in df.groupby('id'):
        a=g[g.sty=='a_thinking'][col]; m=g[g.sty=='my_thought'][col]
        if len(a) and len(m): per.append((m.mean()-a.mean(), len(a)*len(m)/(len(a)+len(m)), g.src.iloc[0]))
    per = np.array([(x,w) for x,w,_ in per]); srcs = None
    est = (per[:,0]*per[:,1]).sum()/per[:,1].sum()
    bs = [ (lambda idx: (per[idx,0]*per[idx,1]).sum()/per[idx,1].sum())(rng.integers(0,len(per),len(per))) for _ in range(2000)]
    return est, np.std(bs), len(per), np.mean(per[:,0]>0), np.mean(per[:,0]<0)
for col in ['ok','cut']:
    for src in ('all','gsm8k','math'):
        sub = d24 if src=='all' else d24[d24.src==src]
        e,se,n,pp,pn = within(sub, col); print(f'{col} {src}: within-problem diff my-a = {100*e:.1f} pts SE {100*se:.1f} (problems {n}; share +{pp:.2f} / -{pn:.2f})')
d24['ltok']=np.log(d24.tok); e,se,n,_,_ = within(d24,'ltok'); print('log tokens diff', round(e,2), 'SE', round(se,2))
```