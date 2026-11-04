```python
# Hypothesis: finished answers are mostly correct; cut ones show some loop pattern at the end. Look at stats and tails.
print(df.groupby('cut')[['correct','tokens']].agg(['mean','count']))
print(df[~df.cut].groupby('source')['tokens'].describe())
print(df.groupby('source')['cut'].mean())
# tail of a few cut texts
for i in [2, 5, 400]:
    t = samples[i]['text']
    print('-----', i, samples[i]['id'], samples[i]['tokens'], 'chars', len(t))
    print(repr(t[-1200:]))
```