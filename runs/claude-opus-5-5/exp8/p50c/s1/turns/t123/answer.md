The remaining failures split into three groups: 52 cut-offs (36 MATH, 16 GSM8K) and 47 wrong finished answers (27 GSM8K, 20 MATH). 27 wrong GSM8K answers out of 128 is surprisingly high for grade-school problems. Some could be grading or format issues (units, decimals, wrong quantity) rather than reasoning errors. So first I'll compare predicted and reference answers for the wrong ones, and check whether the cut texts end in exact loops.

```python
# Hypothesis: some GSM 'wrong' are format/quantity issues; cut ones still mostly exact loops
wr=[(x[0],x[1],x[2]) for x in info if not x[2]['correct'] and not x[2]['cut']]
ansmap={r['id']:r['answer'] for r in E16}
for i,(pid,src,s) in enumerate(wr[:47]):
    print(src[:4], pid[-12:], 'ref=',str(ansmap[pid])[:25], '| pred=', str(s['pred'])[:25], '| tok', s['tokens'])
```