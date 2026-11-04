Picking up after the restart. Fwd_11 holds a teacher-forced pass over two looping windows generated under Edit A, with attention maps for every full-attention layer. Edit A zeroes heads 15.7, 23.5 and 23.0 and is my best so far: +22.7 points paired on S1. Under Edit A the model still loops about 61% of the time; the loops just start later. My hypothesis is that other components take over the copying. These could be full-attention heads that look weak in the original model, or the DeltaNet layers. I first check what is still in the kernel and how the windows are annotated, so the induction-mass analysis uses the same definitions as before. If one or two remaining heads put clearly more attention on the "next token of the previous cycle" in-loop than pre-loop, they become the next candidates. If none does, the copying under Edit A is carried elsewhere.

```python
# Hypothesis check setup: inspect kernel state + window metadata for fwd_11 (edit-A loop windows)
import json, os, numpy as np
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and type(v).__name__ in ('function','dict','list','ndarray','DataFrame','int','float','str')}
print(sorted([k for k,v in g.items() if v=='function'])[:80])
print(sorted([k for k,v in g.items() if v!='function'])[:120])
wins = json.load(open('/work/tf_wins_A.json'))
print(type(wins), len(wins))
if isinstance(wins, list):
    for w in wins: print({k:(v if not isinstance(v,str) else v[:80]) for k,v in w.items() if k!='text'})
else:
    print(list(wins.keys())[:20])
meta = json.load(open('/work/fwd_11/meta.json')); print({k:(str(v)[:200]) for k,v in meta.items()})
```