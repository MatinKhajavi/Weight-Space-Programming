X1 (C plus head 23.4 zeroed) gives exactly the same accuracy as C on the 512 paired items: 60.7% vs 60.7%, difference 0.0 ± 2.0, 53 wrong→right and 53 right→wrong. It reduces cut-offs from 24.8% to 19.5%, but those answers become wrong instead of correct. That is why the current weights are back to plain C, which was the right call. I will record this in the notes. Next, a cheap cross-tab of outcomes (ok/cut/wrong under C versus under X1) to tell two explanations apart. If the right→wrong cases are mostly finished answers turned wrong, head 23.4 is doing real computation. If they are mostly correct answers turned into cut-offs, the churn is just trajectory divergence. This decides whether copying heads with mixed roles can be edited selectively, for example by removing only their retrieval at line starts, instead of being zeroed outright.

```python
# Hypothesis: 23.4 ablation trades loops for wrong answers (computation damage) -- cross-tab outcome categories C vs X1
def cat(v): return 'ok' if v[0] else ('cut' if v[1] else 'wrong')
import collections
ct = collections.Counter((cat(mC[k]), cat(m22[k])) for k in keys)
print('C -> X1 counts:', dict(sorted(ct.items())))
for src in ('gsm8k','math'):
    ct2 = collections.Counter((cat(mC[k]), cat(m22[k])) for k in keys if mC[k][3]==src)
    print(src, dict(sorted(ct2.items())))
with open('/work/notes.md','a') as f:
    f.write("\n- PASS 17 = gen_22 (X1 = C + head 23.4 zeroed, SH512, edited): acc 60.7 vs C 60.7 (paired diff 0.0 SE 2.0, W->R 53 R->W 53); cut 24.8 -> 19.5 but wrongs rose equally. 23.4 NOT worth ablating; reverted to C (current /work/edited = C).\n")
```