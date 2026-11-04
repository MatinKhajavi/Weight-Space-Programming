The saved copies are in .npz format under a .npy name, so `np.load` returned an archive. That does not matter, because E6 is exactly reproduced by its construction formula (earlier check: difference 0.0). I will rebuild E6 from the formula, W₀ − β·Q₆Q₆ᵀW₀ with β = 3, 3, 2 at L15/L19/L23. As a cross-check I will compare the rebuilt tensors with the archive contents, then build the held-out set H as planned.

```python
# Restore E6 via exact formula (cross-check vs saved archive); build fresh held-out set H (160 MATH like MATH-500 + 96 GSM8K).
for l in [15,19,23]:
    nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W0=load_original(nm); bt={15:3.,19:3.,23:2.}[l]
    W6=W0-bt*Q6@(Q6.T@W0); save_edit(nm, W6)
    z=np.load(f'/work/cand_E6/{nm}.npy'); arr=z[z.files[0]]
    print(l, 'max|rebuilt - archived|:', float(np.abs(arr.astype(np.float32)-W6.numpy()).max()), arr.dtype)
used_ids=set()
for f_ in os.listdir('/work'):
    if f_.startswith('gen_') and f_.endswith('.jsonl'):
        for r in readj('/work/'+f_): used_ids.add(r['id'])
for f_ in ['tf3_items.json','tf3_meta_items.json','tf6_set.json','sets.json']:
    try:
        s_=open('/work/'+f_).read(); used_ids.update(re.findall(r'(?:math_train_[a-z_]+_\d+|gsm8k_train_\d+)', s_))
    except Exception as e: print('skip', f_, e)
print('used ids', len(used_ids))
rngH=np.random.default_rng(777)
mrows=[json.loads(l) for l in open('/practice/math_train.jsonl')]; grows=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
quotaH={'Level 1':14,'Level 2':29,'Level 3':34,'Level 4':41,'Level 5':42}; H=[]
for L_,q_ in quotaH.items():
    pool=sorted(o['id'] for o in mrows if o.get('level')==L_ and o['id'] not in used_ids); H+=list(rngH.choice(pool,q_,replace=False))
gpool=sorted(o['id'] for o in grows if o['id'] not in used_ids); H+=list(rngH.choice(gpool,96,replace=False))
H=[str(x) for x in H]; print(len(H), 'overlap with used:', len(set(H)&used_ids))
json.dump({'H':H}, open('/work/sets_H.json','w'))
print(json.dumps(H, separators=(',',':')))
```