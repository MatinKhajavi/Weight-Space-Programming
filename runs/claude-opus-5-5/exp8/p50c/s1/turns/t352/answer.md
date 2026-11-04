On H8 and H9, BW scored 59.2% (GSM 64.5%, MATH 53.9%) with 22.5% cut, against 61.3% for C on the same problems and seeds. Across three comparisons BW comes out slightly below C:

| Set | BW − C (points) |
|---|---|
| SH512 | 0.0 |
| H6 + H7 | −2.3 |
| H8 + H9 | −2.1 |

Pooled, that is about −1.5 ± 1.1 over 1,536 paired items. The brake's Wait-logit component carries nearly all of its effect. The full column's extra indirect route through layer 23, which BR showed lowers Wait on its own, adds a small further benefit. This fits C having the lowest line-initial Wait rate of any variant (1.1 vs 2.1 per 1,000 lines). C stays the submission.

With 5 passes and 4 Python runs left, the safest informative use is to restore C now and spend the remaining passes on further paired original-vs-C confirmations on new fresh sets. This run restores C and checks it against /work/best_final, computes BW's exact paired differences, builds five fresh sets H11–H15 and logs everything.

```python
# R17 hypothesis: none (bookkeeping) - restore C (final), verify vs best_final, quantify BW-C on fresh sets, build fresh sets H11-H15 for the last paired C-vs-original passes.
import os, json, random, collections, numpy as np, torch
for k,v in C_TARGET.items(): save_edit(k,v)
for n in list(edited()):
    if n not in C_TARGET: drop_edit(n)
okC=all((load(k)-v).abs().max().item()<1e-6 for k,v in C_TARGET.items())
okBF=all(np.abs(np.load(f'/work/best_final/{k}.npy')-C_TARGET[k].numpy()).max()<1e-6 for k in C_TARGET)
print('C restored', okC, 'matches best_final', okBF, [n.replace(P,'L') for n in edited()])
def pair(vfile, cfiles):
    V=flat(vfile); Cc=flat(*cfiles); ks=[k for k in V if k in Cc]
    d=np.array([(V[k]=='ok')-(Cc[k]=='ok') for k in ks],float); return d
d1=pair('/work/gen_52.jsonl',['/work/gen_43_edited.jsonl','/work/gen_44_edited.jsonl']); d2=pair('/work/gen_53.jsonl',['/work/gen_48_edited.jsonl','/work/gen_49_edited.jsonl']); d0=pair('/work/gen_46.jsonl',['/work/gen_13.jsonl','/work/gen_16_edited.jsonl'])
dall=np.concatenate([d0,d1,d2]); dfr=np.concatenate([d1,d2])
f=lambda d: f"{d.mean()*100:+.1f} SE {d.std(ddof=1)/np.sqrt(len(d))*100:.1f} (n{len(d)})"
bwline=f"BW-C: SH512 {f(d0)} | H6+H7 {f(d1)} | H8+H9 {f(d2)} | fresh pooled {f(dfr)} | all {f(dall)}"
print(bwline)
used=set()
for fl in glob.glob('/work/gen_*.jsonl'):
    for l in open(fl):
        try: used.add(json.loads(l)['id'])
        except Exception: pass
for fl in glob.glob('/work/*_ids.json'):
    try: used|=set(json.load(open(fl)))
    except Exception: pass
rng=random.Random(4242)
Mfree=collections.defaultdict(list)
for r in M:
    if r['id'] not in used: Mfree[lev(r)].append(r['id'])
Gfree=[r['id'] for r in G if r['id'] not in used]
for L_ in Mfree: rng.shuffle(Mfree[L_])
rng.shuffle(Gfree)
for si,name in enumerate(['H11','H12','H13','H14','H15']):
    ids=[]
    for L_,k in need.items(): ids+=Mfree[L_][si*k:(si+1)*k]
    ids+=Gfree[si*128:(si+1)*128]
    json.dump(ids,open(f'/work/{name}_ids.json','w')); print(f'{name}->', group(ids).replace('math_train_','M:').replace('gsm8k_train','G').replace('counting_and_probability','cp').replace('intermediate_algebra','ia').replace('number_theory','nt').replace('prealgebra','pre').replace('precalculus','pc').replace('geometry','geo').replace('algebra','alg'))
with open('/work/notes.md','a') as fh:
    fh.write(f"- PASS 44 = gen_52 (BW on H6+H7): 60.2 vs C 62.5. PASS 45 = gen_53 (BW on H8+H9): 59.2 vs C 61.3. {bwline} => Wait component carries nearly all of the brake effect; full column's indirect L23 route adds a small extra (C has lowest Wait rate). FINAL stays C (restored+verified {okC}, = best_final {okBF}).\n- Fresh sets H11-H15 created for last paired C-vs-original passes 46-50.\n")
```