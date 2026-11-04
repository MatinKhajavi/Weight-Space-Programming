No record has a truthy `correct` or `grade` field, so the grade key has another name. I will print the keys and the string fields of the first record directly and then locate a finished, correct one. This is still bookkeeping: the goal is to reconstruct the full assistant turn (thinking, </think>, answer) for the benign reference texts.

```python
# Bookkeeping: inspect raw record structure of gen_2.jsonl
print(len(recs2), list(recs2[0].keys()))
for k_, v_ in recs2[0].items():
    if isinstance(v_, str): print(k_, len(v_), repr(v_[:100]), '...', repr(v_[-100:]))
    else: print(k_, repr(v_)[:120])
```