The 19 cut texts without an exact loop are genuine long struggles on hard MATH problems: almost all lines are unique, with doubt and recheck spirals and sometimes deep indentation. I see no single repeated mechanism there. The clearer target is the 34 exact loops, 13% of all samples. To understand how the model enters them, I will print the text just before the loop onset for a few examples. I also want to measure how often the loop line already occurred before the loop started, which would point to line copying, and what precedes each repeat.

```python
# Hypothesis: C loops start when a just-written line is re-emitted (copy from previous occurrence); show entry context
def show_entry(pid, before=500):
    s=[x[2] for x in cut if x[0]==pid][0]; t=s['text']
    r=[x for x in rows if x[0]==pid][0]; p,on=r[2],r[3]
    print('=== ',pid,'P',p,'onset char',on,'of',len(t))
    print(repr(t[on-before:on])[-before:])
    print('--- loop unit:',repr(t[on:on+p])[:200])
for pid in ['math_train_counting_and_probability_708','math_train_number_theory_780','math_train_intermediate_algebra_1074']:
    try: show_entry(pid, 450)
    except Exception as e: print(pid,'ERR',e)
print([r[0] for r in rows if r[2]][:40])
```