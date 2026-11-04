`analyze` returns one row per scored token: text index `b`, repetition index `k` (0 = lead, 1–6 = loop periods), probability of the actual token `p`, and its probability under the evaluation's sampling (`ps`: temperature 0.6, top-p 0.95). Rows are in the same order for all models, so they pair directly. Already visible: at one late line-start token in the last text, the original gives 0.98, E3 0.107, E6 0.001.

I'll now summarise, paired across original, E2, E3 and E6:
- (a) mean p by repetition k;
- (b) per-period escape probability from my earlier `esc` helper;
- (c) p at tokens that are "Wait" openers, by k;
- (d) the paired change in log-probability on the lead tokens (k = 0), the normal reasoning before each loop.

If E6 works as intended, (a)–(c) should collapse at decision points, (d) should be near zero, and the effect should be stronger than E3's.

```python
# Paired summaries: loop-token p by k, escape/period (esc), P(actual) at 'Wait' openers by k, and lead (k=0) dlogprob E6-orig
import pandas as pd
res['E2']=analyze(*TK['E2'], False)
idsA=np.concatenate([ids4[b,1:L4[b]] for b in range(14)])  # actual next tokens aligned with logprob rows? check length
print('rows', len(res['orig']), 'cand aligned len', len(idsA))
tab=pd.DataFrame({nm:res[nm].groupby('k')['p'].mean() for nm in ['orig','E2','E3','E6']}).round(3); print(tab.T)
for nm in ['orig','E2','E3','E6']:
    try: print(nm,'esc', str(esc(res[nm]))[:300])
    except Exception as e: print(nm,'esc ERR',repr(e)[:200])
d0=(np.log(np.clip(res['E6']['p'].values,1e-12,1))-np.log(np.clip(res['orig']['p'].values,1e-12,1)))
k_=res['orig']['k'].values
print('lead (k=0) dlogp E6-orig: mean %.4f median %.4f |d|>1 frac %.3f n=%d'%(d0[k_==0].mean(), np.median(d0[k_==0]), (np.abs(d0[k_==0])>1).mean(), (k_==0).sum()))
print('loop (k>=1) dlogp mean %.3f'%d0[k_>=1].mean())
```