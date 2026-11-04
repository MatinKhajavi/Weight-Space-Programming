With a strict definition of commit failure (an explicit "answer is X" / boxed / "Final Answer" statement of the correct answer), only 25 of 126 cut C answers had solved the problem; the earlier one-third figure came from a looser criterion. Even in those 25, 68% later entertained other final answers. So a perfect commit fix is worth at most about 5 points, and realistically 1–2. Most cut-offs are failed searches. That points at the reasoning dynamics themselves. GSM8K problems are easy enough that failures there are more likely pathological (doubt spirals and loops) than a lack of capability. I will split C's failures by source into cut and wrong, and check whether the cut GSM answers had computed the correct number at all before their loop started.

```python
# Split C failures by source (cut vs wrong) and check if GSM cut answers computed the correct number before loop onset
tab = Counter(); gsm_cut_has = []
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        s = g[i]; src = 'gsm' if i.startswith('gsm') else 'math'
        st = 'ok' if str(s['correct']) == 'True' else ('cut' if str(s['cut']) == 'True' else 'wrong')
        tab[(src, st)] += 1
        if src == 'gsm' and st == 'cut':
            t = s['text']; P, on = find_loop(t); pre = t[:on] if on else t
            a = str(ref[i]).replace(',', '')
            hits = [m.start() for m in re.finditer(rf'(?<![0-9.]){re.escape(a)}(?![0-9])', pre.replace(',', ''))]
            gsm_cut_has.append((len(hits) > 0, len(hits), (hits[0] / len(t)) if hits else None, P is not None))
for src in ['gsm', 'math']:
    n = sum(v for k, v in tab.items() if k[0] == src); print(src, n, {st: f"{tab[(src, st)]} ({tab[(src, st)] / n:.0%})" for st in ['ok', 'cut', 'wrong']})
h = np.array([x[0] for x in gsm_cut_has]); print('GSM cut', len(h), 'answer number appears before onset:', h.sum(), '| exact loop', sum(x[3] for x in gsm_cut_has),
      '| first hit frac quantiles', np.percentile([x[2] for x in gsm_cut_has if x[2] is not None], [25, 50, 75]).round(2))
```