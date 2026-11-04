The diagnosis is stark. In pool D (512 single samples), 467 answers (91%) were cut at the 32,768-token limit and only 44 (8.6%) were correct. Wrong-but-finished answers are almost absent. So the dominant failure is never finishing, presumably because the model gets stuck in a loop inside its thinking. I now want to see what those cut-off texts look like at the end: exact repetition loops, or drift into garbage. I also want to know where looping starts (the kernel has `periods`, `loop_pos` and `stats` from earlier). That tells me which mechanism to go after.

```python
# Hypothesis: cut-offs are repetition loops inside <think>; check where they end, loop periods, and earlier loop stats
cut=[(r,s) for r in rows for s in r['samples'] if s['cut']]
ok=[(r,s) for r in rows for s in r['samples'] if s['correct']]
print('correct tokens mean', np.mean([int(s['tokens']) for r,s in ok]), 'max', max(int(s['tokens']) for r,s in ok))
print('cut with </think>:', sum('</think>' in s['text'] for r,s in cut), 'of', len(cut))
for r,s in cut[:4]:
    print('== ', r['id'], repr(s['text'][-300:]).encode('ascii','replace').decode())
for nm in ['periods','loop_pos','pre_pos','stats','per','early','fracs','loop1','loopR','state','units','fin']:
    v=globals().get(nm); print(nm, type(v).__name__, (repr(v)[:200] if not hasattr(v,'shape') else v.shape))
```