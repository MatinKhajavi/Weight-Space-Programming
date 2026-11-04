E4b scored 34.4% on D (GSM8K 45.8%, MATH 27.5%), against 26.95% for E3 and 10.4% for the original on the same problems and seeds. The cut share fell from 73% to 64.5%, and the finished answers rose from 137 to 188. Twelve of those 188 finished answers are wrong (93.6% correct), against 3 of 137 for E3. That could be mild damage, or simply harder problems now finishing. Now I compute the proper paired statistics against gen_11 (E3) and gen_2 (original), clustered by problem, along with w→r and r→w counts. I will also check whether the 12 wrong finished answers come from problems E3 solved (damage) or from problems E3 never finished (newly reached, harder). A paired E4b−E3 gain above about 2 SE, with the new wrong answers mostly on previously unfinished hard problems, would support keeping E4b.

```python
# Paired analysis: E4b (gen_12) vs E3 (gen_11) vs original (gen_2) on D, clustered SE by problem; check new wrong answers
def load_gen(path):
    out = {}; cnt = collections.Counter()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            pid = r.get('problem') or r.get('id') or r.get('pid')
            key = (pid, cnt[pid]); cnt[pid] += 1
            out[key] = r
    return out
G12 = load_gen('/work/gen_12.jsonl')
r0 = next(iter(G12.values())); print(list(r0.keys())[:15])
def paired(Ga, Gb, label):
    keys = sorted(set(Ga) & set(Gb))
    byp = collections.defaultdict(list)
    for k in keys:
        byp[k[0]].append(float(Ga[k]['correct']) - float(Gb[k]['correct']))
    for src in ('all','gsm8k','math'):
        ps = [p for p in byp if src=='all' or p.startswith(src)]
        d = np.array([np.mean(byp[p]) for p in ps])
        wr = sum(1 for k in keys if (src=='all' or k[0].startswith(src)) and Ga[k]['correct'] and not Gb[k]['correct'])
        rw = sum(1 for k in keys if (src=='all' or k[0].startswith(src)) and not Ga[k]['correct'] and Gb[k]['correct'])
        print(f'{label} {src}: n_prob={len(ps)} diff={100*d.mean():+.2f} SE={100*d.std(ddof=1)/np.sqrt(len(d)):.2f}  w->r={wr} r->w={rw}')
print(len(G12), len(G11), len(G2))
paired(G12, G11, 'E4b-E3')
paired(G12, G2, 'E4b-orig')
paired(G11, G2, 'E3-orig')
# finished stats
for name, G in (('orig',G2),('E3',G11),('E4b',G12)):
    fin = [k for k in G if not G[k]['cut']]
    wrong = [k for k in fin if not G[k]['correct']]
    print(name, 'finished', len(fin), 'wrong among finished', len(wrong), 'mean tokens fin', int(np.mean([G[k]['tokens'] for k in fin])))
wrong12 = [k for k in G12 if not G12[k]['cut'] and not G12[k]['correct']]
print('E4b wrong-finished: E3 status (cut/correct):', [(k[0][-22:], G11[k]['cut'], G11[k]['correct']) for k in wrong12])
```