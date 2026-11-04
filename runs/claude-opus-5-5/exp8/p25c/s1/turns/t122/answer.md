Each record holds the problem id, source, answer and a list of samples, and the kernel still has my earlier helpers (`paired`, `load_gen2`, `loop_info`, `categorize`, `loop_opener`, `line_openers`). Before computing anything, I will print the sample-level keys and the source of those helpers, so that I reuse exactly the same pairing, loop-detection and categorisation logic as for E3/E4b/E5. Then I can compare like with like. This block only inspects; no hypothesis is tested yet.

```python
# Inspect helper source + sample schema so paired/loop analyses are identical to earlier ones.
import inspect
print(list(g16[0]['samples'][0].keys()))
for fn in ['paired','load_gen2','loop_info','categorize','loop_opener','line_openers']:
    src = inspect.getsource(globals()[fn])
    print('-----', fn, len(src)); print(src[:1200])
```