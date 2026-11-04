The helpers have no retrievable source; they were defined via exec. I'll read their signatures and call them on one known E6 cut-off text to see what they return. If that's unclear, I'll write small fresh helpers instead. Either way, the goal is a loop/noloop breakdown of E6's cut-offs plus short tails of the noloop ones.

```python
# Inspect helper signatures and test find_loop / cat_of / near_rep on an E6 cut-off text
for f in ['find_loop','loop_info','cat_of','categorize','near_rep','paired','pooled','load_gen','acc_','cut_','think_end']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as e: print(f, e)
E6files={'D':'gen_16.jsonl','H':'gen_18_edited.jsonl','H2':'gen_21_edited.jsonl'}
E6rows={k:[json.loads(l) for l in open('/work/'+fn)] for k,fn in E6files.items()}
ex=[s for r in E6rows['D'] for s in r['samples'] if s['cut']][0]
for f in ['find_loop','loop_info','near_rep','cat_of']:
    try:
        out=globals()[f](ex['text']); print(f, str(out)[:300])
    except Exception as e: print(f, 'ERR', str(e)[:200])
```