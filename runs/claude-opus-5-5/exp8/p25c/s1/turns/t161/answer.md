I'm continuing after the restart. The current edit is E6. It flips the component of the attention output that lies along a rank-8 "line-opener" subspace (Wait, So, Let, Okay, If, I, This, But) in the o_proj of layers 15, 19 and 23. On two held-out sets it gave a paired gain of +31 points. I have 3 forward passes and 164 Python runs left. Before spending a pass I need to recover what I was doing (the last output looks like loop windows from the original model) and check which `max_tokens` the earlier generation passes used. If they used fewer than the evaluation's 32,768 tokens, part of the "cut" category could behave differently in the real evaluation. That decides between a final confirmation at full length and one more candidate edit.

```python
# Reorient: read the start of the notes, list kernel variables, check max_tokens/lengths in earlier generation passes
import json, os, numpy as np
notes = open('/work/notes.md').read()
head = notes[:max(0, len(notes)-8000)]
print(len(notes), len(head))
print(head[:5200])
print('-----VARS-----')
vs = [(k, type(v).__name__) for k, v in globals().items() if not k.startswith('_') and type(v).__name__ not in ('module','function','type','builtin_function_or_method')]
print(len(vs)); print(', '.join(f"{k}:{t}" for k, t in vs[-80:]))
```