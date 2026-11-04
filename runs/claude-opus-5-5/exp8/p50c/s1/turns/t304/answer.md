The "Ready" check gives no actionable evidence. C's correct answers use "Ready" less often than the original's (26% vs 58% of texts), consistent with the doubled brake's small off-target pull on that token. But they close fine with alternatives ("Okay, ready to write.", "Final check complete."). C's cut answers never reach a closing stage at all (0% contain "Ready"), so this side effect cannot explain the remaining failures. Projecting it out of the brake would be a cosmetic refinement with a near-zero expected gain, so I will not spend a pass on it. All the mechanism-level leads are now tested. Ablating the copy heads and doubling the Wait brake are both specific and necessary. The other candidate levers (further copy heads, the Maybe brake, repetition-tracking neurons, deep nesting, opening style, formatting) are either refuted or symptoms of being stuck on hard problems. I will therefore use the remaining passes to measure the final edit C precisely on fresh, never-used problems with paired original-vs-C runs, separately for each benchmark. This run builds H3 (256 fresh GSM8K) and H4 (256 fresh MATH at MATH-500's level mix), excludes every problem used before, and prints the IDs compactly.

```python
# Build fresh held-out sets for final confirmation: H3 = 256 unused GSM8K, H4 = 256 unused MATH with MATH-500 level mix (22/46/54/66/68)
used=set()
for fn in os.listdir('/work'):
    if fn.endswith('_ids.json'):
        try: used|=set(json.load(open('/work/'+fn)))
        except Exception: pass
for fn in os.listdir('/work'):
    if fn.startswith('gen_') and fn.endswith('.jsonl'):
        for r in lg('/work/'+fn): used.add(r['id'])
print('used ids', len(used))
gsm=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]; mth=[json.loads(l) for l in open('/practice/math_train.jsonl')]
rng=random.Random(2024)
g_av=[r['id'] for r in gsm if r['id'] not in used]; H3=rng.sample(g_av,256)
H4=[]
for lv,k in zip([1,2,3,4,5],[22,46,54,66,68]):
    av=[r['id'] for r in mth if r['id'] not in used and str(r.get('level','')).strip().endswith(str(lv))]
    H4+=rng.sample(av,k)
print(len(H3),len(H4),len(set(H3)|set(H4)), 'example level field', mth[0].get('level'))
json.dump(H3,open('/work/H3_ids.json','w')); json.dump(H4,open('/work/H4_ids.json','w'))
for nm,S in (('H3',H3),('H4',H4)):
    grp=collections.OrderedDict()
    for i in S: pre,num=i.rsplit('_',1); grp.setdefault(pre,[]).append(num)
    for pre,nums in grp.items(): print(nm, pre, len(nums), ' '.join(nums))
```