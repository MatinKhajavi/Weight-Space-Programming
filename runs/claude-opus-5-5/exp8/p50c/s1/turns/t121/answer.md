The held-out confirmation pass (H1, 256 problems not used to choose anything) shows that the combined edit generalizes. Edit A ablates the copy/induction heads 15.7, 23.5 and 23.0. The "brake" edit doubles the down-projection column of neuron 22.573, which suppresses line-initial "Wait". Together they raise accuracy from 12.1% to 61.3%, a paired difference of +49.2 points with SE 3.2: 127 problems went wrong-to-right and 1 right-to-wrong. Cut-offs fell from 89% to 21%. So the two loop mechanisms I identified, line-start copying through induction heads and runaway "Wait" lines, generalize beyond the selection set. 52 cut-off answers and 47 wrong answers remain. Before spending more passes, I will record this result and characterize the remaining failures in the edited H1 outputs: exact loops vs long non-looping reasoning vs wrong answers. That tells me which mechanism to go after next. If most remaining cut-offs are still exact periodic loops, another copy/loop driver is the target. If not, the remaining failures are more likely reasoning errors that targeted edits cannot easily reach.

```python
# Hypothesis: remaining failures of edit A+brake on H1 are mostly exact loops (another copy driver) -- check categories
import json, re, numpy as np, collections
with open('/work/notes.md','a') as f:
    f.write("\n- PASS 13 = gen_16 (both, H1 held-out, A+brake x2): orig 12.1% (gsm 16.4, math 7.8) cut 89% | edited 61.3% (gsm 66.4, math 56.3) cut 20.7% wrong 47; diff +49.2 SE 3.2, W->R 127 R->W 1. CONFIRMED on held-out. /work/edited = A+brake x2 (=best_C_x2).\n")
E16 = [json.loads(l) for l in open('/work/gen_16_edited.jsonl')]
O16 = [json.loads(l) for l in open('/work/gen_16_original.jsonl')]
print(len(E16), list(E16[0].keys()))
r=E16[0]; print({k:(str(v)[:80]) for k,v in r.items() if k!='text'})
```