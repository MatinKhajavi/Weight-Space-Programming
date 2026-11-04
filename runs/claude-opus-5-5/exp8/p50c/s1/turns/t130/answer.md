Several helpers from the earlier analyses survive: dla, dla_diff, neuron_dla, indm/ind_targets for induction mass, set_heads/snapshot for edits, and paired for comparing generations. I'll print their signatures and first lines to reuse them correctly rather than rewriting them.

```python
# Hypothesis: reuse earlier helpers; print their signatures/docs compactly
import inspect
for fn in ['set_heads','snapshot','paired','readgen','dla','dla_diff','neuron_dla','indm','ind_targets','find_loop','wrate','summarize','prompt_of']:
    try:
        src=inspect.getsource(globals()[fn]).split('\n')
        print('##',fn, '|', ' / '.join(l.strip() for l in src[:4])[:300])
    except Exception as e: print(fn,'ERR',e)
```