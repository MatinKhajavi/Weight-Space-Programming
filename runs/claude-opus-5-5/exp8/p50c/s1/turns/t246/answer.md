The closing sections are the clearest difference between outcomes:

| Phrase | Correct texts | Cut texts |
|---|---|---|
| "Final Answer" | 86% | 9% |
| "Final check" | 83% | 12% |
| "Double check" | 80% | 10% |
| "Ready" | 26% | 1% |

So cut texts almost never reach the wrap-up phase of the model's outline-style thinking. The last line before `</think>` begins with a bullet, "Final" or "The" far more often than with "Ready", so the brake's side effect on "Ready" and `</think>` only acts at a point the cut texts never reach. That repair is low value, and I will set it aside.

Two failure modes remain unmeasured and are cheap to check:
- **Missing answer:** how many non-cut answers under C end without a `\boxed{}` answer.
- **Opening style:** whether the opening style of the thinking predicts success. The model starts either with a Gemini-like "Here is my thought process… 1. **Analyze the Request:**" outline or with other openings. If one style loops far more, the style switch at the start of thinking is a mechanism worth localizing.

If accuracy differs strongly by opening, I will pursue it; if not, I will return to the loop-copying circuitry.

```python
# Hypothesis check: (a) non-cut answers without a boxed answer; (b) does the opening style of thinking predict outcome under C?
rows = []
for r in recC:
    s = r['samples'][0]; t = s['text']
    op = t.lstrip()[:28]
    rows.append(dict(src=r['source'], ok=s['correct'], cut=s['cut'], pred=s['pred'], op=op, tok=s['tokens']))
df = pd.DataFrame(rows) if 'pd' in globals() else None
import pandas as pd
df = pd.DataFrame(rows)
print('non-cut no-pred:', int(((~df.cut) & df.pred.isna()).sum()), 'of', len(df), '| non-cut wrong with pred:', int(((~df.cut)&(~df.ok)&df.pred.notna()).sum()))
df['style'] = df.op.str.slice(0,12)
g = df.groupby('style').agg(n=('ok','size'), acc=('ok','mean'), cut=('cut','mean')).sort_values('n', ascending=False).head(10)
g.index = [A(i) for i in g.index]; print(g.round(2))
```