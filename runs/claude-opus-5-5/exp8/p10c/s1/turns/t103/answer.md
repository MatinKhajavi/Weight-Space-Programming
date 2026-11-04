Edit v3 is a clear success on the measure that matters:

| Comparison | Accuracy | Paired difference | Wrong→right / right→wrong |
|---|---|---|---|
| setA, v3 vs v2 | 47.3% vs 38.7% | +8.6 (SE 2.9) | 39 / 17 |
| diagnostic problems, v3 vs original | 48.4% vs 9.4% | +39.1 (SE 3.1) | 100 / 0 |

The mechanism prediction also came true:
- Exact loops on setA fell from 144 to 82.
- Median lock-in onset moved from 8.5k to 15.8k tokens.
- Finished answers rose from 104 to 144.

So the late copy heads in layers 19 and 23 are causally responsible for much of the repetition lock-in.

One warning sign: accuracy among finished answers on setA fell from 0.94 to 0.84 (0.92 on the diagnostic problems). That could be selection, since newly finishing problems are the harder long-thinking ones, or real damage to copying. To separate the two, I will compare v2 and v3 on the problems that finished under both edits. A drop there would indicate a capability cost that limits further dose increases.

I will also examine the cut answers that do not end in an exact loop (30 on setA, 32 on the diagnostic problems). I want to know whether they are paraphrased near-loops or genuine long reasoning; that decides the next target. Finally, I will save v3 as best_3 and log the result.

```python
# Hypothesis: the drop in accuracy among finished (0.94->0.84) is mostly selection (newly finished = harder problems), not damage; check commonly-finished subset and characterize non-exact cuts
shutil.copytree('/work/edited', '/work/best_3', dirs_exist_ok=True)
with open('/work/notes.md', 'a') as f:
    f.write("\n## Pass 5 RESULT (gen_10, edited v3, setA+diagS): v3 acc 47.85% (512). setA v3-v2 paired +8.6 SE 2.9 (39 w->r, 17 r->w); setA v3-orig +39.1; diagS v3-orig 48.4 vs 9.4 = +39.1 SE 3.1 (100 w->r, 0 r->w). setA exact loops 144->82, onset median 8.5k->15.8k, finished 104->144, acc among finished 0.94->0.84 (diagS 0.92). CURRENT BEST = v3 (/work/best_3).\n")
both_fin = [i for i in setA if not r9e[i]['cut'] and not r10[i]['cut']]
only3 = [i for i in setA if r9e[i]['cut'] and not r10[i]['cut']]; only2 = [i for i in setA if not r9e[i]['cut'] and r10[i]['cut']]
print(f"setA finished in both: {len(both_fin)} | acc v2 {np.mean([r9e[i]['correct'] for i in both_fin]):.3f} v3 {np.mean([r10[i]['correct'] for i in both_fin]):.3f}")
print(f"newly finished in v3: {len(only3)} acc {np.mean([r10[i]['correct'] for i in only3]):.3f} | lost (finished v2, cut v3): {len(only2)} (v2 acc on them {np.mean([r9e[i]['correct'] for i in only2]):.3f})")
print(f"v3 finished length median: both {np.median([r10[i]['tokens'] for i in both_fin]):.0f}, newly {np.median([r10[i]['tokens'] for i in only3]):.0f}")
lvl = {rr_['id']: rr_.get('level') for rr_ in mt}
wrong_fin = [i for i in setA + diagS if not r10[i]['cut'] and not r10[i]['correct']]
print('v3 finished-but-wrong:', len(wrong_fin), '| levels', collections.Counter(lvl.get(i, 'gsm') for i in wrong_fin).most_common(), '| median tokens', np.median([r10[i]['tokens'] for i in wrong_fin]))
def ngram_rep(ids, n=20, tail=6000):
    x = ids[-tail:]; grams = [tuple(x[k:k+n]) for k in range(0, len(x)-n)]
    c = collections.Counter(grams); return sum(v for v in c.values() if v > 1) / max(1, len(grams))
nonex = []
for i in setA + diagS:
    s = r10[i]
    if s['cut']:
        ids = toks(s['text']); on, p = tail_loop(ids)
        if on is None: nonex.append((i, ngram_rep(ids)))
reps = np.array([r for _, r in nonex])
print(f"v3 non-exact cuts: {len(nonex)} | share of repeated 20-grams in last 6k tokens: median {np.median(reps):.2f}, >0.5: {(reps>0.5).sum()}, <0.1: {(reps<0.1).sum()}")
for i, rp in nonex[:4]:
    print('--', i, f'rep {rp:.2f} |', r10[i]['text'][-300:].replace('\n', ' | '))
```