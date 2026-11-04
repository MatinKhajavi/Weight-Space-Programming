The last pass tested v7, which is v4 with the halving of the early copy heads L11H1 and L15H7 undone. It is a necessity test for the edit with the weakest evidence. v7 scored 53.5% against 57.2% for v4 on the same 512 problems. Its cut-off share rose from 27% to 34% (174 cut answers against 140). So removing that edit appears to cost about 3.7 points, mainly through more unfinished loops, as the copy-head hypothesis predicts. Next I will compute the paired statistics (v7 against v4, same seeds) and check how the kernel's edit state differs from the stored v4 copy in /work/best_4. Then I will restore v4 exactly, because v7's results say v4 is the better choice. A paired drop clearly larger than its standard error, with the extra failures in cut-off loops, would confirm that the L11/L15 halvings are causally needed.

```python
# Hypothesis: v7 (L11H1/L15H7 restored) is worse than v4 paired on the same 512 -> keep those edits; first inspect file formats and best_4 contents.
import json, os, glob, numpy as np
def readj(p):
    return [json.loads(l) for l in open(p)]
g14 = readj('/work/gen_14.jsonl'); g11 = readj('/work/gen_11.jsonl')
print(len(g14), len(g11), list(g14[0].keys()))
print({k: (str(v)[:60] if not isinstance(v,str) else v[:60]) for k,v in g14[0].items() if k not in ('text','completion','output')})
print(sorted(os.listdir('/work/best_4'))[:20], sorted(os.listdir('/work/edited'))[:20])
print(edited())
```