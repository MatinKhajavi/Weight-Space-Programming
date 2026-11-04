```python
# Hypothesis: there may be undocumented helpers for building forward requests from files; inspect kernel globals.
print([k for k in globals().keys() if not k.startswith('_')][:80])
import inspect
for name in ['load','save_edit','drop_edit','edited','load_original']:
    try:
        print(name, inspect.signature(globals()[name]))
        print(inspect.getsource(globals()[name])[:600])
    except Exception as e:
        print(name, 'ERR', e)
```