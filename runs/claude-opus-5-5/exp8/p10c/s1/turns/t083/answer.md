The helper sources are not retrievable, but I know their signatures. So I will call `toks`, `loop_info`, `period_of`, `onset_period`, `ans_before` and `analyze` on one cut v2 sample and one finished v2 sample to see what they return. Then I can use them for the full loop characterization without re-implementing anything. If the return values are compact tuples or dicts (onset, period, cycle stats), I will build the v2 loop census on top of them next.

```python
# Hypothesis: (bookkeeping) learn the return formats of my earlier helpers by calling them on one cut and one finished v2 sample
g9e = readjl('/work/gen_9_edited.jsonl')
cut_s = next(s for rr in g9e for s in rr['samples'] if s['cut'])
fin_rec = next(rr for rr in g9e if not rr['samples'][0]['cut'])
ids_c = toks(cut_s['text']); print('toks type', type(ids_c).__name__, len(ids_c))
def short(x):
    s = repr(x); return s[:400]
for name, call in [('loop_info', lambda: loop_info(ids_c)), ('period_of', lambda: period_of(ids_c)), ('onset_period', lambda: onset_period(ids_c)),
                   ('line_start_positions', lambda: line_start_positions(ids_c)[:10]), ('wait_rate', lambda: wait_rate(ids_c, 0, 3000)),
                   ('finish_stats', lambda: finish_stats(cut_s['text']))]:
    try: print(name, '->', short(call()))
    except Exception as e: print(name, 'ERR', e)
try:
    op = onset_period(ids_c); print('ans_before ->', short(ans_before(ids_c, op[0] if isinstance(op,(tuple,list)) else op, g9e[0]['answer'])))
except Exception as e: print('ans_before ERR', e)
try: print('analyze ->', short(analyze(g9e[:3])))
except Exception as e: print('analyze ERR', e)
```