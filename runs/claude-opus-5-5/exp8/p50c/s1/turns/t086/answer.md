The attribution narrows things down. Under Edit A, the extra in-loop push toward the copied token still comes mainly from the layer-23 mixer (+2.45 and +2.44 in the two windows), then the layer-19 mixer (+1.2 and +0.9), the layer-23 MLP (+0.9 and +0.4) and the layer-15 mixer (+0.5 and +0.3). So the remaining copying runs through the other heads of the same full-attention layers, which Edit B showed are needed for correct computation. Cutting more copy heads is a dead end, so I'll go after the other large failure category. Under Edit A the model often reaches the correct answer and then keeps doubting, never committing, until it loops (GSM: 67% of cut answers contained the correct answer before the loop). I'll reread several of these Edit-A transcripts from gen_6 at the point where the answer first appears. I want to see what the model does next: re-verifying, reinterpreting the question, or recomputing. That shows whether a specific, identifiable decision point (such as "answer stated → Wait") is the bottleneck.

```python
# Diagnose "failure to commit": in edit-A GSM cut samples, what follows the first appearance of the correct answer?
import re
g6e = [json.loads(l) for l in open('/work/gen_6_edited.jsonl')]
print(list(g6e[0].keys()))
r0 = g6e[0]; print({k:(str(v)[:80]) for k,v in r0.items() if k not in ('text','output','completion')})
```