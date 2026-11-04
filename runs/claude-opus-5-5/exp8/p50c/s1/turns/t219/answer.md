C's wrong GSM8K answers are genuine reasoning errors, not formatting problems, and they are long (median 15k tokens on grade-school problems). Strikingly, many are off by exactly a factor of two (12 vs 6, 180 vs 90, 60 vs 30, 40 vs 20, 36 vs 72, 16 vs 8, 150 vs 75, 4 vs 2). That pattern fits a story where the model reached a correct reading and then talked itself into a reinterpretation, such as double-counting or "each/per pair", during a long doubt episode. To test that "over-doubting" story, I will check in each wrong GSM8K answer whether and where the reference number first appears as a computed result ("= X"), compared with where the final wrong answer first appears. I will also show where the switch happened in a few examples.

```python
# Over-doubting test: in C's wrong GSM answers, was the correct number computed first and later abandoned for the wrong one?
def first_result_pos(t, num):
    num = str(num).replace(',', '').replace('\\$', '').strip()
    m = re.search(rf'=\s*\\?\$?\s*{re.escape(num)}(?![0-9.])', t.replace(',', ''))
    return m.start() / len(t) if m else None
had = []
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        s = g[i]
        if i.startswith('gsm') and str(s['correct']) != 'True' and str(s['cut']) != 'True':
            t = s['text']; k = t.find('</think>'); th = t[:k] if k > 0 else t
            pr = re.sub(r'[^0-9.\-]', '', str(s['pred'])) or 'NA'
            had.append((i, first_result_pos(th, ref[i]), first_result_pos(th, pr)))
ref_first = [h for h in had if h[1] is not None and (h[2] is None or h[1] < h[2])]
print('wrong GSM', len(had), '| ref result computed at all', sum(h[1] is not None for h in had), '| ref computed BEFORE wrong final', len(ref_first))
print('  ref first-pos quantiles', np.percentile([h[1] for h in had if h[1] is not None], [25, 50, 75]).round(2),
      '| wrong-final first-pos quantiles', np.percentile([h[2] for h in had if h[2] is not None], [25, 50, 75]).round(2))
for i, rp, pp in ref_first[:3]:
    t = (g13.get(i) or g16e[i])['text'] if i in g13 else g16e[i]['text']
    j = int(rp * len(t)); print(i[-5:], 'ref', ref[i], '@', round(rp, 2), 'pred@', pp and round(pp, 2), '|', repr(t[j - 120:j + 100]).encode('ascii', 'replace').decode())
```