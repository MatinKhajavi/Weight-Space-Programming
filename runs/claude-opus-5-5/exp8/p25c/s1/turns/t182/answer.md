E6's leftover loops have no single trigger. They start early (median at 27% of the text, about 8–9k tokens), then fill the rest of the budget, and they are problem-specific. Some are two-line oscillations between competing expressions ("$x_2 = 2x - 4$? No."), some restate a fact, some are doubt questions or arithmetic re-checks. They come from the same copy machinery that coherent reasoning uses (E8). So I see no further targeted lever I could justify and test with 2 passes, and E6 remains the final edit.

The most useful remaining measurement is a paired side-effect profile. On normal, non-looping reasoning text, how much does E6 change the model's next-token predictions, and where? A targeted edit should change little except line-opener decisions in repetitive contexts. If an earlier teacher-forced pass of the original model on non-loop texts exists, a single E6 pass on the same texts gives that profile. Next I'll check what the stored teacher-forced passes contain (model, edits, number of texts).

```python
# Which stored TF passes ran the ORIGINAL model on reusable texts (incl. non-loop reasoning)? -> allows one-pass paired side-effect profile of E6
for d_ in ['fwd_4','fwd_6','fwd_9','fwd_10','fwd_13','fwd_15']:
    try:
        m=json.load(open(f'/work/{d_}/meta.json'))
        L_=m['lengths']; L_=json.loads(L_) if isinstance(L_,str) else L_
        fl=m['files']; fl=fl if isinstance(fl,dict) else {}
        print(d_, m['model'], str(m['edited_tensors'])[:60], 'ntexts', len(L_), 'tok', sum(L_), 'files', [k for k in (fl.keys() if isinstance(fl,dict) else [])][:6])
    except Exception as e: print(d_, 'ERR', e)
t6=json.load(open('/work/tf6_set.json')); print('tf6_set', type(t6), len(t6), (list(t6[0].keys()) if isinstance(t6,list) and isinstance(t6[0],dict) else list(t6.keys())[:10] if isinstance(t6,dict) else ''))
```