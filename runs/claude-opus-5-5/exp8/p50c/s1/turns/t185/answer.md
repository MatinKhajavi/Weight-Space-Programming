V2 changes almost nothing. On S1 it reaches 60.55% (GSM8K 73.4, MATH 47.7) with 27.3% cut and 31 wrong, against edit C's 60.2%, 28.9% cut and 28 wrong on the same seeds. So removing the brake's suppression of readiness tokens does not measurably help. Before discarding the idea I want the paired view and the mechanism check. Paired view: right↔wrong transitions and the paired difference with its standard error. Mechanism check: did " Ready" / "Final" lines become more frequent, and did post-solution verification get shorter? If readiness lines rose but outcomes did not move, the brake's side effect was real but is not the bottleneck for concluding. If readiness lines did not rise, the neuron is probably inactive at conclusion points.

```python
# Paired analysis V2 (gen_21) vs C (gen_13) on S1: transitions, paired diff, readiness-line rates, lengths
g21=loadj('/work/gen_21.jsonl')
def okv(G_): return {r_['id']:int(r_['samples'][0]['correct']) for r_ in G_}
oC=okv(gC13); oV=okv(g21); ids_=list(oC)
d_=np.array([oV[i]-oC[i] for i in ids_])
print('V2-C paired diff', round(100*d_.mean(),2), 'SE', round(100*d_.std(ddof=1)/np.sqrt(len(d_)),2), 'W->R', int((d_>0).sum()), 'R->W', int((d_<0).sum()))
mV=stmap(g21); print(pd.crosstab(pd.Series([mC[i] for i in ids_],name='C'), pd.Series([mV[i] for i in ids_],name='V2')))
rV=rate_table([r_['samples'][0] for r_ in g21])[0]; rCs=rate_table([r_['samples'][0] for r_ in gC13])[0]
print('line-initial per 1k tok  C vs V2:', {w:(round(rCs.get(w,0),2), round(rV.get(w,0),2)) for w in ['Ready','Final','Done','Result','Wait','Maybe','So','Okay','Looks']})
tokC=[r_['samples'][0]['tokens'] for r_ in gC13 if r_['samples'][0]['correct']]; tokV=[r_['samples'][0]['tokens'] for r_ in g21 if r_['samples'][0]['correct']]
print('correct-answer tokens median C', np.median(tokC), 'V2', np.median(tokV))
```