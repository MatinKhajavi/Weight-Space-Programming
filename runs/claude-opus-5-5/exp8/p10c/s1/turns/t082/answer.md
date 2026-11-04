The records hold their generations in a `samples` list, and the kernel still has my earlier helpers: `loop_info`, `period_of`, `onset_period`, `analyze`, `ans_before`, `wait_rate` and `line_start_positions`. Rather than rewrite loop detection, I will print the sample fields and the helper signatures and sources so I can reuse them exactly as before. This keeps the v2 loop analysis comparable with the numbers already in my notes, which were computed with these same functions. The check works if the sources are retrievable and the sample fields include text, grade and length.

```python
# Hypothesis: (bookkeeping) inspect sample fields and helper signatures/sources to reuse the earlier loop analysis consistently
import inspect
print('sample keys:', {k:(len(v) if isinstance(v,(str,list)) and len(str(v))>80 else v) for k,v in r['samples'][0].items()})
for fn in ['readjl','toks','loop_info','period_of','onset_period','line_start_positions','analyze','ans_before','wait_rate','finish_stats','mass']:
    f = globals()[fn]
    try: src = inspect.getsource(f)
    except Exception as e: src = 'NA '+str(e)
    print('-----', fn, str(inspect.signature(f)))
    print(src[:900])
```