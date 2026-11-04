My earlier helpers survive in the kernel: `find_loop`, `categorize`, `near_rep`, `paired`, `pooled`, among others. Rather than rewrite them, I'll print the source of the key ones so I use them correctly. These are `find_loop`, `loop_info`, `categorize`/`cat_of`, `near_rep`, `paired` and `pooled`. That fixes the exact definitions of "exact loop", "near-loop" and "noloop" behind the categories in my notes.

```python
# Recover definitions of my loop/category/pairing helpers to reuse them consistently
import inspect
for f in ['find_loop','loop_info','cat_of','categorize','near_rep','paired','pooled']:
    try:
        s=inspect.getsource(globals()[f]); print(s[:900]); print('---')
    except Exception as e:
        print(f, 'no source', e)
```