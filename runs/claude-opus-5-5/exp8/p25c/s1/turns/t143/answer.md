The held-out loops still contain E6-set openers in about a quarter of their lines (So 9.5%, I'll 6.3%, This 4.2%, I 3.8%, If 2%). The rest are math lines (24.5%) and non-set words (Is, No, Maybe, Because, Why). The loop-free cuts are brute-force enumerations ("$1525^2 = ...$, $1526^2 = ...$"), counter near-loops ("$h_{1114} = 0.13$..."), and a few wrap-up near-loops such as "So the answer is 169. I'll write the response." that never emit </think>. The wrong answers are mostly coherent but mistaken solutions on harder items. None of this points to a new single component. Enumeration and counter loops are strategy failures, and a closing fix would have to raise </think> where the indented-outline format competes with it, which is close to the out-of-scope "close early" knob. I will therefore treat E6 as the final edit family and use the remaining passes for evidence. The first piece is a dense collateral-damage check on normal reasoning text: per-token log-probability change on reference solutions under E6. To decide whether that needs one pass or two, I will check whether an earlier teacher-forced pass of the original model on reference solutions exists to pair with.

```python
# Check which earlier TF passes exist (original on reference solutions?) to pair a dense E6 collateral-damage measurement.
for d_ in sorted([f_ for f_ in os.listdir('/work') if f_.startswith('fwd_')], key=lambda x:int(x.split('_')[1])):
    try:
        mt=json.load(open(f'/work/{d_}/meta.json'))
        keys=list(mt.keys()) if isinstance(mt,dict) else []
        print(d_, {k:(str(mt[k])[:90]) for k in keys if k not in ('texts','lengths','offsets')}, '| files', sorted(os.listdir(f'/work/{d_}'))[:6])
    except Exception as e: print(d_, 'ERR', e)
for f_ in ['tf6_set.json','tfW_texts.json','tf12_texts.json']:
    s_=json.load(open('/work/'+f_)); print(f_, type(s_).__name__, len(s_), str(s_)[:200].encode('ascii','replace').decode())
```