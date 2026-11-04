Two observations stand out. First, the opening style depends on the weights, not only on the seed. The original model on the S1 seeds opens with "my thought process" 67 times, "a thinking process" 154 times and other openings 35 times. The brake alone (gen_15) leaves this unchanged. Ablating the L23 induction heads shifts it: 54/196/6 under A and C, and 26/227/3 under B, which removes more L23 heads.

So my head ablation pushed generations toward the failure-prone "a thinking process" mode, partly offsetting its own benefit. That tells me L23 heads take part in choosing the opening. Second, I have no multi-sample runs, so a within-problem causal test needs a new pass. Before designing it, I will check that the style–outcome association also holds in the original model's runs, gen_3 and gen_16_original, and see what the "other" openings are. If the original also shows "my thought process" doing much better at the same source and level, the mode effect is a property of the network, not of my edits.

```python
# Hypothesis: the style-outcome association is intrinsic (also in the ORIGINAL model), and 'other' openings are a third mode
def style_table(paths):
    rows=[]
    for p in paths:
        for r in readj(p):
            s=r['samples'][0]; rows.append(dict(src=r['source'], sty=sty(s['text'].lstrip()[:28]), ok=s['correct'], cut=s['cut'], op=s['text'].lstrip()[:20], lvl=mt[r['id']]['level'] if r['id'] in mt else 'gsm'))
    return pd.DataFrame(rows)
do = style_table(['/work/gen_3.jsonl','/work/gen_16_original.jsonl'])
print(do.groupby(['src','sty']).agg(n=('ok','size'), acc=('ok','mean'), cut=('cut','mean')).round(2))
print(do[do.src=='math'].groupby(['lvl','sty']).ok.agg(['size','mean']).round(2).unstack())
print('other openings:', collections.Counter(A(o)[:14] for o in do[do.sty=='other'].op).most_common(8))
```