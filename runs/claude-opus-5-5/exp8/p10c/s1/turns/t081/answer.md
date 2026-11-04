Pass 4 tested v2 (v1 plus halving the two strongest copy heads, L11H1 and L15H7) on the same 256 setA problems. Generation is deterministic per problem: the original texts were identical across passes. So v2 can be paired directly with v1. v2 beat v1 by +5.5 points (SE 2.9; 35 problems wrong→right, 21 right→wrong), giving about 38.7% against the original's 8.2%. The gain came almost entirely from cut-off answers turning into finished ones: finished rose from 86 to 104 and stayed 94% correct.

The largest failure is still exact periodic loops: 144 of 256 answers end in one, with median onset about 8.6k tokens. Almost every correct answer comes from a finished sample, so further gains must come from breaking these loops.

Before choosing the next intervention I will record this result, save v2 as best_2, and list the helper functions left in the kernel so I can reuse my loop-detection code. Next I will characterize what the remaining v2 loop cycles are made of: other doubt words such as "But" or "Actually", restated answers, or repeated computations. If the cycles are dominated by a different doubt word or a small set of line-start tokens, that points to another neuron-level driver. If they are mostly doubt-free restatements, the copy circuit is the better target.

```python
# Hypothesis: (bookkeeping) record pass-4 result, save v2 as best_2, and list helper functions/record keys from earlier runs for reuse
import json, shutil, os
with open('/work/notes.md','a') as f:
    f.write("\n## Pass 4 RESULT (gen_9, both, setA 256): v2 vs v1 paired (same seeds, original texts identical across passes) +5.5 pts SE 2.9 (35 w->r, 21 r->w); v2 acc ~38.7% vs original 8.2%. cut 170->152, finished 86->104 (acc among finished 0.94), exact loops 143->144, median onset 8.99k->8.56k. CURRENT BEST = v2 (copy in /work/best_2).\n")
if not os.path.exists('/work/best_2'): shutil.copytree('/work/edited','/work/best_2')
fns = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__']
print('functions:', fns)
r = json.loads(open('/work/gen_9_edited.jsonl').readline())
print({k:(len(v) if isinstance(v,(str,list)) and len(str(v))>80 else v) for k,v in r.items()})
print('edited tensors:', edited())
```