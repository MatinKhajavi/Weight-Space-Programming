The test strongly supports the self-reinforcement picture. The copy heads' attention to earlier periods rises steadily with the repetition index: layer 23 head 5 (induction) goes 0.39→0.70, 23.6 (same-token) 0.57→0.81, 23.4 0.25→0.40, 23.0 0.19→0.30, 23.7 0.24→0.37, 23.2 0.11→0.26. In the middle layers, 15.7 rises 0.52→0.75, 11.1 0.48→0.60 and 11.3 (same-token) 0.30→0.56. Each new copy of the period pulls these heads' attention further onto earlier copies, which fits the loop-token probability climbing to 0.99. Once a token passes roughly 0.85–0.9 raw probability, temperature 0.6 plus top-p 0.95 removes all alternatives, so the loop becomes absorbing. The next step is a causal test in one teacher-forced pass, which needs reusable data first. I will build a larger set of loop excerpts and a "benign" set of the model's own normal reasoning, so one pass per edit can measure both loop escape and collateral damage. Here I recover the helper functions and check text lengths to size that set within the token budget.

```python
# Bookkeeping for building a larger TF set: recover helper sources, loop table columns, text lengths
import inspect
for fn in ['rep_index','loop_info','prompt_str','ntok','tok_ids','dla']:
    try:
        src = inspect.getsource(globals()[fn]); print(src[:700]); print('---')
    except Exception as e:
        print(fn, 'no source:', e)
print('c2 cols', c2.columns.tolist()); print(c2.head(2).to_string()[:800])
print('cut cols', cut.columns.tolist())
ln = [ntok(t) for t in fin_txt]
print('fin_txt tokens: n', len(ln), 'quantiles', np.percentile(ln,[0,25,50,75,90,100]).astype(int))
print('compact[0] keys/type', type(compact[0]), (list(compact[0].keys()) if isinstance(compact[0],dict) else len(compact[0])))
```