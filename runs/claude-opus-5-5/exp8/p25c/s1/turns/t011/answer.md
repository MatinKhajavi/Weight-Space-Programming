```python
# Analyze gen_2: lengths, cut, correctness, and look at text tails
g2 = readjsonl('/work/gen_2.jsonl')
rows = []
for r in g2:
    for j,s in enumerate(r['samples']):
        rows.append(dict(id=r['id'], src=r['source'], j=j, tokens=s['tokens'], cut=s['cut'], correct=s['correct'], pred=s['pred'], text=s['text']))
import pandas as pd
df = pd.DataFrame(rows)
print(df.groupby('src')[['cut','correct']].mean())
print(df[~df.cut].tokens.describe())
print((df.text.str.contains('</think>')).mean())
# check fraction containing think close
fin = df[~df.cut]
print(len(fin), fin.correct.mean())
print(fin.groupby('src').size())
```