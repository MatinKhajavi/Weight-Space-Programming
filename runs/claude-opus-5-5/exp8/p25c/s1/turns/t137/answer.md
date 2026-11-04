Under E6 the E5 wrap-up loops mostly became other failures (8 wrong, 7 correct, the rest other loops), not correct finishes. Overall there is large two-way churn: 75 of 213 E5-correct samples turned non-correct and about 78 non-correct samples turned correct. Whether a long trajectory falls into a loop is close to random once one token differs, so E5 and E6 are statistically indistinguishable (+1.0 ± 2.6). The real decision now is not E5 versus E6 but whether the edit family holds up on problems never used for selection. I will restore E6 as the current edit from its saved copy, its D-set result being nominally best, and draw a fresh held-out set H. H has 160 MATH problems stratified by level like MATH-500 (14/29/34/41/42 for L1–L5) and 96 GSM8K problems, excluding every id used in earlier passes. One sample per problem maximises the number of distinct problems for the paired "both" comparison. The confirmation succeeds if E6 − original on H is many SEs above zero, as on D (+32.8 ± 2.7); a much smaller gain would suggest overfitting to D.

```python
# Restore E6 as current edit; build fresh held-out set H (160 MATH stratified like MATH-500 + 96 GSM8K, unused ids).
for f_ in os.listdir('/work/cand_E6'):
    nm=f_[:-4]; save_edit(nm, torch.from_numpy(np.load(os.path.join('/work/cand_E6',f_))).float())
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W0=load_original(nm); bt={15:3.,19:3.,23:2.}[l]
    print(l, 'E6 restored, diff to formula:', float((load(nm)-(W0-bt*Q6@(Q6.T@W0))).abs().max()))
used_ids=set()
for f_ in os.listdir('/work'):
    if f_.startswith('gen_') and f_.endswith('.jsonl'):
        for r in readj('/work/'+f_): used_ids.add(r['id'])
for f_ in ['tf3_items.json','tf3_meta_items.json','tf6_set.json','sets.json']:
    try:
        s_=open('/work/'+f_).read(); used_ids.update(re.findall(r'(?:math_train_[a-z_]+_\d+|gsm8k_train_\d+|aime_[\w\-]+)', s_))
    except Exception as e: print('skip', f_, e)
print('used ids', len(used_ids))
rngH=np.random.default_rng(777)
mrows=[json.loads(l) for l in open('/practice/math_train.jsonl')]; grows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
quotaH={'Level 1':14,'Level 2':29,'Level 3':34,'Level 4':41,'Level 5':42}; H=[]
for L_,q_ in quotaH.items():
    pool=[o['id'] for o in mrows if o.get('level')==L_ and o['id'] not in used_ids]; H+=list(rngH.choice(pool,q_,replace=False))
gpool=[o['id'] for o in grows if o['id'] not in used_ids]; H+=list(rngH.choice(gpool,96,replace=False))
H=[str(x) for x in H]; print(len(H), len(set(H)&used_ids))
json.dump({'H':H}, open('/work/sets_H.json','w'))
print(json.dumps(H, separators=(',',':')))
```