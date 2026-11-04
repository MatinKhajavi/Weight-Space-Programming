The opening sentence of the thinking is strongly associated with the outcome under C:

| Opening | n | Accuracy | Cut |
|---|---|---|---|
| "Here's my thought process" | 80 | 94% | 4% |
| "Here is my thought process" | 21 | 86% | 10% |
| "Here's a thinking process" | 311 | 59% | 22% |
| "Here is a thinking process" | 89 | 35% | 52% |

Also, every non-cut wrong answer has a boxed prediction, so "finishing without an answer" does not occur. The opening pattern may well be a confound: the opening could depend on problem type or difficulty, for example "my thought process" for easy GSM items. So I will first control for source and MATH level. If the gap survives within the same source and level, the opening probably selects a persistent reasoning "mode", and that would be worth testing causally with several samples per problem. If it vanishes, the opening is just a difficulty marker.

```python
# Hypothesis: opening style ("my thought process" vs "a thinking process") predicts outcome even within source/level (not just difficulty)
mt = {j['id']: j for j in (json.loads(l) for l in open('/practice/math_train.jsonl'))}
df['id'] = [r['id'] for r in recC]
df['lvl'] = [mt[i]['level'] if i in mt else 0 for i in df.id]
def sty(op):
    o = op.lower()
    if 'my thought' in o: return 'my_thought'
    if 'a thinking' in o: return 'a_thinking'
    if 'a thought' in o: return 'a_thought'
    if 'my thinking' in o: return 'my_thinking'
    return 'other'
df['sty'] = df.op.map(sty)
print(pd.crosstab([df.src], df.sty))
print(df.groupby(['src','sty']).ok.agg(['size','mean']).round(2).unstack())
print(df[df.src=='math'].groupby(['lvl','sty']).ok.agg(['size','mean']).round(2).unstack())
```