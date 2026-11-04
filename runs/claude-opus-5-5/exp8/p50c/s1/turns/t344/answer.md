The corrected checks give two clear answers.

1. **Brake:** with the zero-centred final-norm gain, the remainder used in BR has a direct Wait-logit effect of 0.001 against 0.310 for the full column (0.3%). Its effect, cutting line-initial Wait from 356 to 44 per 1,000 lines, therefore runs entirely through layer 23. The brake is cleanly a "Wait" suppressor with a direct path and an indirect path through layer 23.
2. **Copy heads:** heads 23.0 and 23.5 do not copy token identity through their direct path. Their centred own-token copy z-scores are −0.26 and +0.16, and no rank of removal changes that meaningfully. The rebuilt CS1 would remove only two directions (99% of each head's norm kept), so it would equal leaving the heads intact. I will not spend a pass on it.

So these heads support loops through their attention to the matching position in the previous cycle, as measured earlier, while what they write back is not aligned with token directions.

The mechanistic work is complete and C stays the final edit. With 6 Python runs and 11 passes left, this run:
- restores and verifies C and saves a final copy to /work/best_final;
- builds three more fresh held-out sets (H8–H10, each 128 MATH problems at the MATH-500 level mix plus 128 GSM8K);
- selects 128 AIME problems for a generalisation check on much harder problems.

These need no further edits, only paired passes.

```python
# R15 hypothesis: none (bookkeeping) - restore/verify C as final, save copy, build fresh held-out sets H8-H10 and an AIME set for the remaining paired confirmation passes.
import shutil, json, glob, random, collections, os
for k,v in C_TARGET.items(): save_edit(k,v)
for n in list(edited()):
    if n not in C_TARGET: drop_edit(n)
okC=all((load(k)-v).abs().max().item()<1e-6 for k,v in C_TARGET.items())
if os.path.exists('/work/best_final'): shutil.rmtree('/work/best_final')
shutil.copytree('/work/edited','/work/best_final')
print('C restored', okC, [n.replace(P,'L') for n in edited()], '| best_final files', os.listdir('/work/best_final'))
used=set()
for f in glob.glob('/work/gen_*.jsonl'):
    for l in open(f):
        try: used.add(json.loads(l)['id'])
        except Exception: pass
for f in glob.glob('/work/*_ids.json'):
    try: used|=set(json.load(open(f)))
    except Exception: pass
M=[json.loads(l) for l in open('/practice/math_train.jsonl')]; G=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]; AI=[json.loads(l) for l in open('/practice/aime.jsonl')]
rng=random.Random(777)
Mfree=collections.defaultdict(list)
for r in M:
    if r['id'] not in used: Mfree[lev(r)].append(r['id'])
Gfree=[r['id'] for r in G if r['id'] not in used]
for L_ in Mfree: rng.shuffle(Mfree[L_])
rng.shuffle(Gfree)
need={1:11,2:23,3:27,4:33,5:34}
for si,name in enumerate(['H8','H9','H10']):
    ids=[]
    for L_,k in need.items(): ids+=Mfree[L_][si*k:(si+1)*k]
    ids+=Gfree[si*128:(si+1)*128]
    json.dump(ids,open(f'/work/{name}_ids.json','w')); print(f'{name} ({len(ids)}) ->', group(ids))
af=[r['id'] for r in AI if r['id'] not in used]; rng.shuffle(af); aset=af[:128]
json.dump(aset,open('/work/AIME128_ids.json','w')); print('AIME128 ->', group(aset))
with open('/work/notes.md','a') as fh:
    fh.write(f"- Corrected-gain recheck: brake remainder direct Wait effect 0.001 vs col 0.310 (0.3%) => BR's Wait reduction is indirect via L23. Copy-subspace (centered, gain 1+w): heads 23.0/23.5 own-token copy z = -0.26/+0.16 => NOT direct token copiers; their loop role is via induction-target attention with non-token-aligned outputs. CS1 not run (would be ~= heads intact).\n- FINAL = C, restored+verified ({okC}), copy in /work/best_final. Fresh sets H8-H10 + AIME128 created for remaining paired passes.\n")
```