The picture has changed qualitatively. Under E5 only 133 of the 268 cut answers end in an exact verbatim loop (E4b: 300 of 330; E3: 351 of 375). The remaining exact loops are spread thinly over many openers (' Could' 84, ' Then' 48, 'This' 42, ' So' 41, ' Is', ' I', ' But', ...), so no single opener dominates anymore. The other half of the cut answers run out the 32,768-token budget without exact repetition. They could be near-loops with drifting details, very long cycles, enumeration (one excerpt is long division "30467 / 3 = 10155"), or genuinely slow but progressing reasoning. Each would need a different lever. I measure, for every non-exact cut answer, how much of the final 2,000 characters repeats material from earlier: the share of 40-character shingles of the tail that already occurred in the preceding 30,000 characters. I also read a few tails.

```python
# Classify E5 cut answers that are NOT exact loops: near-repetition share of the tail, and short tail excerpts
def near_rep(t, tail=2000, back=30000, w=40, step=10):
    tl = t[-tail:]; prev = t[-(tail+back):-tail]
    if len(prev) < w: return 0.0
    S = set(prev[i:i+w] for i in range(0, len(prev)-w, 3))
    sh = [tl[i:i+w] for i in range(0, len(tl)-w, step)]
    return float(np.mean([s in S for s in sh]))
nonexact = [x for x in LT14 if x[1] is None]
nr = [(x[0], near_rep(G14[x[0]]['text'])) for x in nonexact]
vals = np.array([v for _, v in nr])
print('non-exact cut:', len(nonexact), ' near-rep share quantiles 10/25/50/75/90:', np.round(np.quantile(vals, [.1,.25,.5,.75,.9]), 2))
print('share with near-rep > 0.5:', float(np.mean(vals > 0.5)), ' < 0.2:', float(np.mean(vals < 0.2)))
# examples: low near-rep (progressing?) and high near-rep (near-loop)
lo = sorted(nr, key=lambda z: z[1])[:3]; hi = sorted(nr, key=lambda z: -z[1])[:2]; mid = sorted(nr, key=lambda z: abs(z[1]-0.5))[:2]
for tag, grp in (('LOW', lo), ('MID', mid), ('HIGH', hi)):
    for k, v in grp:
        print('---', tag, k, round(v,2))
        print(asc(G14[k]['text'][-260:]).replace('\n','|'))
```