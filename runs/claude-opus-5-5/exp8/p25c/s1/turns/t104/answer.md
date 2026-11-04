E5 reaches 42.2% on D (GSM8K 53.1%, MATH 35.6%), against 34.4% for E4b, 27.0% for E3 and 10.4% for the original on the same seeds. The cut share fell from 64% to 52%. On the other hand, 29 finished answers are wrong and 2 finished without an answer (12 and 0 under E4b). I now compute the paired E5−E4b difference with problem-clustered SE, plus E5 vs original. I also check where the new wrong finished answers come from: problems E4b solved on the same seed (damage from the L23 edit), or problems E4b never finished (newly reached, harder). I also compare finished-answer accuracy by source and level. A paired gain well above 2 SE, with the new errors mostly on previously cut answers, would make E5 the new best.

```python
# Paired analysis E5 (gen_14) vs E4b (gen_12), E3, original; origin of new wrong-finished answers
G14 = load_gen2('/work/gen_14.jsonl')
paired(G14, G12, 'E5-E4b')
paired(G14, G2, 'E5-orig')
lvl = {}
for x in mth:
    lvl[x['id']] = x.get('level')
for name, G in (('orig',G2),('E3',G11),('E4b',G12),('E5',G14)):
    fin = [k for k in G if not G[k]['cut']]
    wrong = [k for k in fin if not G[k]['correct']]
    print(f"{name}: finished {len(fin)} wrong-finished {len(wrong)} fin-acc {1-len(wrong)/max(1,len(fin)):.3f} "
          f"| fin-acc gsm {np.mean([G[k]['correct'] for k in fin if k[0].startswith('gsm')]):.3f} math {np.mean([G[k]['correct'] for k in fin if k[0].startswith('math')]):.3f}")
wrong14 = [k for k in G14 if not G14[k]['cut'] and not G14[k]['correct']]
st = collections.Counter()
for k in wrong14:
    s4 = 'E4b-cut' if G12[k]['cut'] else ('E4b-correct' if G12[k]['correct'] else 'E4b-wrongfinished')
    st[s4] += 1
print('E5 wrong-finished, status under E4b on same seed:', dict(st))
print('E5 wrong-finished where E4b correct:', [(k[0][-26:], k[1], G14[k]['pred'][:20], G12[k]['pred'][:20]) for k in wrong14 if (not G12[k]['cut']) and G12[k]['correct']])
# accuracy by MATH level for each model
for name, G in (('orig',G2),('E4b',G12),('E5',G14)):
    byl = collections.defaultdict(list)
    for k, r in G.items():
        if k[0].startswith('math'): byl[lvl.get(k[0])].append(r['correct'])
    print(name, {l: round(float(np.mean(v)),2) for l, v in sorted(byl.items(), key=lambda z: str(z[0]))})
```