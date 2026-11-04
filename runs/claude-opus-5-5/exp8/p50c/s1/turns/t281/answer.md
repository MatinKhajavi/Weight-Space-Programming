Pass 22 confirmed edit C on a third held-out set (H2, sampled at MATH-500's level mix plus GSM8K): 59.4% against 10.9% for the original. The paired difference is +48.4 points (SE 3.3), with 127 answers going from wrong to right and 3 from right to wrong. That matches H1 (+49.2), so the repair holds up. The original model fails mainly by looping until the token limit. C fixes most of that with two mechanisms: it zeroes the line-start Wait-copying heads 15.7, 23.0 and 23.5, and it doubles the built-in Wait-brake neuron 22.573. Under C, 21% of answers are still cut off and 20% are wrong, and these are the remaining targets. Python runs (49 left) are now scarcer than forward passes (28 left). So I will first record this result, re-read the early part of my notes that the restart hid (definitions of variants A and B and earlier dead ends), and list the helper functions still in the kernel. That way I don't rebuild tools or repeat refuted experiments.

```python
# Bookkeeping: record pass 22, recover early notes (hidden by restart) and list helper functions still in the kernel
import json, re, os
with open('/work/notes.md') as f: notes = f.read()
with open('/work/notes.md','a') as f:
    f.write("\n- PASS 22 = gen_27 (both, H2 fresh held-out, C): orig 10.9% (gsm 14.1, math 7.8) cut 90% | C 59.4% (gsm 65.6, math 53.1) cut 20.7% wrong 50 noans 1; diff +48.4 SE 3.3, W->R 127 R->W 3. C CONFIRMED on 2nd fresh held-out set.\n")
print(len(notes))
head = notes[:max(0, len(notes)-8000)]
print(head[:6500])
helpers = [k for k,v in list(globals().items()) if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__']
print(helpers)
```