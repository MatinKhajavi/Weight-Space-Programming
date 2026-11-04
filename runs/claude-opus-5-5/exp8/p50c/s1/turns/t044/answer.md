The cut-offs are exact verbatim loops. One line, almost always beginning with "Wait," (for example "Wait, I'll write `\boxed{2}`."), repeats until the 32k limit. Only 1 of 470 cut texts ever reached `</think>`. Correct answers average about 6.9k tokens. So repetition inside the thinking is by far the largest failure (91% of samples), and that is what I'll go after. Before designing an edit I want three things: where loops start (`fracs`, `periods`), what fwd_4 contained, and what the run-36 table measured (`a_pre`/`a_loop`). That table showed the loop-versus-pre gap opening at the full-attention layers (after layer 15, then at the top).

```python
# Hypothesis: loops start well before the limit and have short periods; recover what fwd_4 / a_pre / a_loop were
print('fracs (loop onset frac?)', np.percentile(fracs,[10,25,50,75,90]).round(2))
pv=[p for p in periods if p]; print('periods n',len(pv),'pct',np.percentile(pv,[10,25,50,75,90]), 'None',sum(p is None for p in periods))
m4=json.load(open('/work/fwd_4/meta.json')); print({k:(str(v)[:300]) for k,v in m4.items()})
for nm in ['a_pre','a_loop','d','dd','d2','dm','A','Ah','E','F','X','Xc','y','yc','P','P1','P2','sel','res','top','ind','idx','r2','w','z','u']:
    v=globals().get(nm); print(nm, type(v).__name__, getattr(v,'shape', (repr(v)[:150])))
```