The early notes fill in the history. Ablating more copying heads than 15.7/23.0/23.5 (variant B) hurt computation, and making those three heads negative (-0.5) hurt as well, so C is the best head set found. Before choosing the next intervention I need to know the current state of the failures. Under C about 20% of answers are cut and 20% are wrong, and earlier evidence says roughly a third of the cut answers had already found the answer before looping. Two candidate targets remain:
- a failure to commit after solving;
- residual line-start copying (head 23.4 ablation cut loops but raised wrong answers equally).

To pick between them I will reuse the helper functions already in the kernel. In this run I only print their source so I call them correctly, plus the record format of the generation files. This avoids wasting scarce Python runs on signature errors.

```python
# Recall helper APIs (to reuse rather than rewrite): print source of key helpers and the gen record format
import inspect, json
for fn in ['readgen','load_gen','find_loop','onset_of','ans_in','first_ans_idx','cat','set_heads','snapshot','paired','commit_follow','first_correct_stmt']:
    try:
        src = inspect.getsource(globals()[fn])
        print('=== ', fn, len(src)); print(src[:700])
    except Exception as e:
        print('ERR', fn, e)
r0 = json.loads(open('/work/gen_27_edited.jsonl').readline())
print({k:(type(v).__name__, (v[:60] if isinstance(v,str) else v) if k!='text' else len(v)) for k,v in r0.items()})
```