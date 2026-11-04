The generation file stores one line per problem, with a 'samples' list, rather than one line per answer. G11 and G2 were parsed into (problem, sample index) records with fields src/text/pred/correct/tokens/cut. I will parse gen_12 the same way, after checking which field names the samples use, and then rerun the paired comparison. Nothing has changed in the hypothesis being tested.

```python
# Fix loader for per-problem lines with 'samples'; rerun paired comparison E4b vs E3 vs original
with open('/work/gen_12.jsonl') as fh:
    first = json.loads(fh.readline())
print(list(first['samples'][0].keys()))
def load_gen2(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            for i, s in enumerate(r['samples']):
                rec = dict(s); rec['src'] = r['source']
                if 'correct' not in rec:
                    for kk in ('grade','is_correct','score'):
                        if kk in rec: rec['correct'] = bool(rec[kk])
                if 'cut' not in rec:
                    for kk in ('truncated','cut_at_max','hit_max'):
                        if kk in rec: rec['cut'] = bool(rec[kk])
                if 'tokens' not in rec:
                    for kk in ('n_tokens','length','num_tokens'):
                        if kk in rec: rec['tokens'] = rec[kk]
                out[(r['id'], i)] = rec
    return out
G12 = load_gen2('/work/gen_12.jsonl')
r0 = next(iter(G12.values())); print({k: (str(v)[:30]) for k, v in r0.items() if k != 'text'})
print(len(G12), sum(1 for k in G12 if k in G11))
paired(G12, G11, 'E4b-E3')
paired(G12, G2, 'E4b-orig')
paired(G11, G2, 'E3-orig')
for name, G in (('orig',G2),('E3',G11),('E4b',G12)):
    fin = [k for k in G if not G[k]['cut']]
    wrong = [k for k in fin if not G[k]['correct']]
    print(name, 'finished', len(fin), 'wrong among finished', len(wrong), 'mean tokens fin', int(np.mean([G[k]['tokens'] for k in fin])))
wrong12 = [k for k in G12 if not G12[k]['cut'] and not G12[k]['correct']]
print('E4b wrong-finished: E3 status (cut,correct):', [(k[0][-24:], k[1], G11[k]['cut'], G11[k]['correct']) for k in wrong12])
```