gen_21 matches C on 'Wait' rate and accuracy, but line-initial 'Maybe' dropped from 2.67 to 1.70 per 1k tokens. So the undocumented 16th pass tested some 'Maybe'-suppressing variant and found no accuracy gain (60.5% vs 60.2%, 18 vs 17 discordant problems). I conclude that the re-expressed doubt in 'Maybe' lines is not the bottleneck, and I will record that. Exact loops are still the largest single failure under C (82 of 126 cut answers), so I want a sharper picture of what the remaining loop units are. Candidates are repeating digit expansions, line-level reasoning cycles, or repeated final-answer statements. I will detect the period and onset for every cut C answer on S1 and H1, and tabulate unit types with short excerpts.

```python
# Categorize C's remaining exact loops (S1 gen_13 + H1 gen_16_edited): period, onset, unit type -> pick next mechanism target
g16e = load_gen('/work/gen_16_edited.jsonl')
def find_loop(t, min_rep=3, maxP=4000):
    tail = t[-12000:]
    n = len(tail)
    for P in range(5, min(maxP, n // min_rep)):
        if tail[-P*min_rep:] == tail[-P*min_rep - P: -P] if n >= P*(min_rep+1) else False:
            # extend backwards to find onset
            k = len(t) - 1
            i = len(t) - P - 1
            while i >= 0 and t[i] == t[i + P]:
                i -= 1
            return P, i + 1
    return None, None
cuts = []
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        s = g[i]
        if str(s['cut']) == 'True':
            P, on = find_loop(s['text'])
            cuts.append((name, i, P, on, len(s['text'])))
print('cuts', len(cuts), 'exact loops', sum(c[2] is not None for c in cuts))
Ps = [c[2] for c in cuts if c[2]]; print('period chars quantiles', np.percentile(Ps, [10, 25, 50, 75, 90]).round())
cat = Counter()
exs = []
for name, i, P, on, L in cuts:
    if P is None: cat['no_exact_loop'] += 1; continue
    t = (g13 if name == 'S1' else g16e)[i]['text']; unit = t[on:on + P]
    if P < 40: c = 'short<40'
    elif '\\boxed' in unit or 'answer is' in unit.lower() or 'final answer' in unit.lower(): c = 'answer_stmt'
    elif unit.count('\n') == 0: c = 'inline_no_newline'
    elif re.search(r'(?m)^\s*(?:\*\s+)?Maybe', unit): c = 'maybe_line'
    else: c = 'multiline_reasoning'
    cat[c] += 1; exs.append((c, name, i, P, round(on / L, 2), unit))
print(cat)
import random; random.seed(0)
for c, name, i, P, frac, unit in random.sample(exs, 14):
    print(c, i[:28], 'P', P, 'onset', frac, '|', repr(unit[:150]).encode('ascii', 'replace').decode())
```