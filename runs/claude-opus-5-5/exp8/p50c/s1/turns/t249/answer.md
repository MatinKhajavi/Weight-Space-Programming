The two openings mark different reasoning modes, not just phrasing. Both continue with the same "1. **Analyze the Request:**" outline, but "a thinking process (that leads to the solution)" texts are much longer and more deeply nested:

| | Median tokens, all answers | Median tokens, correct answers | Lines indented ≥ 8 spaces |
|---|---|---|---|
| GSM, "a thinking" | 8.1k | 5.5k | 74% |
| GSM, "my thought" | 4.1k | 4.0k | 26% |
| MATH, "a thinking" | 24.1k | 11.3k | 41% |
| MATH, "my thought" | 6.5k | 6.1k | 13% |

That fits my earlier finding that deep nesting predicts loops. If the mode causally drives the outcome, it would be the largest remaining lever. But the opening could also reflect the model's own early judgement that a problem is hard. The test is within-problem variation: several samples of the same problem with different openings. Before spending a pass, I will check whether an earlier run (gen_3, the diagnostic run) already has several samples per problem, which would allow a within-problem comparison at no cost.

```python
# Check whether existing runs (esp. gen_3 diagnostic) have multiple samples per problem -> within-problem test of opening style
import glob, os
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    R = readj(p); ns = collections.Counter(len(r['samples']) for r in R)
    ops = collections.Counter(sty(s['text'].lstrip()[:28]) for r in R for s in r['samples'])
    print(os.path.basename(p), 'items', len(R), 'n/item', dict(ns), 'styles', dict(ops))
```