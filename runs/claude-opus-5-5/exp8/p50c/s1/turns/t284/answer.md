The cross-tabs show that ablating 23.4, 11.1 or 23.7 mostly reshuffles outcomes. For example, X1 moves answers from cut to wrong (cut→ok 35 vs ok→cut 20, but ok→wrong 33 vs wrong→ok 18). The ok→wrong counts (30–36 per 512) are about the churn I would expect from any perturbation that changes trajectories (the neutral "Maybe" variant had about 35 per 512 each way). So whole-head ablation is a dead end. C's wrong answers on H2 are genuinely wrong, not formatting slips. A new, cleaner way to look for a loop repair: inside an exact loop the text of every cycle is identical, so any neuron whose activation changes systematically from cycle 1 to cycle k is responding to the repetition itself. Such a neuron would be part of a built-in repetition detector. If some of these neurons push away from the copied token (a built-in loop escape that is too weak), amplifying them would be a targeted, justified repair. The C loop windows already saved in fwd_18 should contain MLP activations, so this needs no new forward pass. This run checks what fwd_18 and its window file contain.

```python
# Check what fwd_18 (C loop windows) contains, to test for built-in 'repetition detector' neurons without a new pass
import os, json, numpy as np
d='/work/fwd_18'
print(sorted(os.listdir(d))[:60])
meta=json.load(open(d+'/meta.json'))
print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<300 else str(v)[:300]) for k,v in meta.items()})
W=json.load(open('/work/tf_wins_C.json'))
print(type(W), len(W))
if isinstance(W,list):
    for w in W: print({k:(v if not isinstance(v,str) else (len(v), v[:80])) for k,v in (w.items() if isinstance(w,dict) else enumerate([w]))})
elif isinstance(W,dict): print({k:(type(v).__name__, (v if not isinstance(v,(str,list)) else len(v))) for k,v in W.items()})
for f in sorted(os.listdir(d)):
    if f.endswith('.npy'):
        a=np.load(os.path.join(d,f), mmap_mode='r'); print(f, a.shape, a.dtype)
```