The generation files store a list of samples per problem. /work/best_4 holds all six v4 tensors, while the kernel's current edit (v7) lacks the layer-11 and layer-15 o_proj edits. Now I will compute the paired v7-vs-v4 comparison on the 512 shared problems. I will also break down where the extra failures went (cut-off loops versus wrong finished answers). Then I will restore v4 from best_4 and check every tensor against the intended edit pattern: columns scaled relative to the original weights, and nothing else changed. If v7 is clearly worse in the paired comparison and the losses are mostly new loops, that confirms the early copy-head edits are doing what we claimed.

```python
# Hypothesis: v7 - v4 paired is clearly negative (losses via more loops) -> restore v4 exactly and verify the edit pattern vs original.
import torch, ast
s0 = g14[0]['samples']
s0 = ast.literal_eval(s0) if isinstance(s0, str) else s0
print(type(s0), list(s0[0].keys()))
def samp(r):
    s = r['samples']; s = ast.literal_eval(s) if isinstance(s, str) else s
    return s[0]
k_corr = [k for k in s0[0] if 'correct' in k or 'grade' in k]; print('grade keys', k_corr)
```